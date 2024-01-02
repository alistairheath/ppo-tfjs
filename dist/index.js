import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
function log(...args) {
    console.log('[PPO]', ...args);
}
class BaseCallback {
    constructor() {
        this.nCalls = 0;
    }
    async _onStep(alg) { return true; }
    async onStep(alg) {
        this.nCalls += 1;
        return this._onStep(alg);
    }
    async _onTrainingStart(alg) { }
    async onTrainingStart(alg) {
        this._onTrainingStart(alg);
    }
    async _onTrainingEnd(alg) { }
    async onTrainingEnd(alg) {
        this._onTrainingEnd(alg);
    }
    async _onRolloutStart(alg) { }
    async onRolloutStart(alg) {
        this._onRolloutStart(alg);
    }
    async _onRolloutEnd(alg) { }
    async onRolloutEnd(alg) {
        this._onRolloutEnd(alg);
    }
}
class FunctionalCallback extends BaseCallback {
    constructor(callback) {
        super();
        this.callback = callback;
    }
    async _onStep(alg) {
        if (this.callback) {
            return this.callback(alg);
        }
        return true;
    }
}
class DictCallback extends BaseCallback {
    constructor(callback) {
        super();
        this.callback = callback;
    }
    async _onStep(alg) {
        if (this.callback && this.callback.onStep) {
            return this.callback.onStep(alg);
        }
        return true;
    }
    async _onTrainingStart(alg) {
        if (this.callback && this.callback.onTrainingStart) {
            this.callback.onTrainingStart(alg);
        }
    }
    async _onTrainingEnd(alg) {
        if (this.callback && this.callback.onTrainingEnd) {
            this.callback.onTrainingEnd(alg);
        }
    }
    async _onRolloutStart(alg) {
        if (this.callback && this.callback.onRolloutStart) {
            this.callback.onRolloutStart(alg);
        }
    }
    async _onRolloutEnd(alg) {
        if (this.callback && this.callback.onRolloutEnd) {
            this.callback.onRolloutEnd(alg);
        }
    }
}
class Buffer {
    constructor(bufferConfig) {
        this.observationBuffer = [];
        this.actionBuffer = [];
        this.rewardBuffer = [];
        this.valueBuffer = [];
        this.logprobabilityBuffer = [];
        this.advantageBuffer = [];
        this.returnBuffer = [];
        this.trajectoryStartIndex = 0;
        this.pointer = 0;
        const bufferConfigDefault = {
            gamma: 0.99,
            lam: 0.95
        };
        this.bufferConfig = { ...bufferConfigDefault, ...bufferConfig };
        this.gamma = this.bufferConfig.gamma;
        this.lam = this.bufferConfig.lam;
        this.reset();
    }
    add(observation, action, reward, value, logprobability) {
        this.observationBuffer.push(observation);
        this.actionBuffer.push(action);
        this.rewardBuffer.push(reward);
        this.valueBuffer.push(value);
        this.logprobabilityBuffer.push(logprobability);
        this.pointer += 1;
    }
    discountedCumulativeSums(arr, coeff) {
        let res = [];
        let s = 0;
        arr.reverse().forEach(v => {
            s = v + s * coeff;
            res.push(s);
        });
        return res.reverse();
    }
    finishTrajectory(lastValue) {
        const rewards = this.rewardBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue * this.gamma);
        const values = this.valueBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue);
        const deltas = rewards
            .slice(0, -1)
            .map((reward, ri) => reward - (values[ri] - this.gamma * values[ri + 1]));
        this.advantageBuffer = this.advantageBuffer
            .concat(this.discountedCumulativeSums(deltas, this.gamma * this.lam));
        this.returnBuffer = this.returnBuffer
            .concat(this.discountedCumulativeSums(rewards, this.gamma).slice(0, -1));
        this.trajectoryStartIndex = this.pointer;
    }
    get() {
        let advantageMean, advantageStd;
        [advantageMean, advantageStd] = tf.tidy(() => [
            tf.mean(this.advantageBuffer).arraySync(),
            tf.moments(this.advantageBuffer).variance.sqrt().arraySync()
        ]);
        this.advantageBuffer = this.advantageBuffer
            .map(advantage => {
            return (advantage - advantageMean) / advantageStd;
        });
        return [
            this.observationBuffer,
            this.actionBuffer,
            this.advantageBuffer,
            this.returnBuffer,
            this.logprobabilityBuffer
        ];
    }
    reset() {
        this.observationBuffer = [];
        this.actionBuffer = [];
        this.advantageBuffer = [];
        this.rewardBuffer = [];
        this.returnBuffer = [];
        this.valueBuffer = [];
        this.logprobabilityBuffer = [];
        this.trajectoryStartIndex = 0;
        this.pointer = 0;
    }
}
export class PPO {
    constructor(env, config) {
        this.randomSeed = 0;
        const configDefault = {
            nSteps: 512,
            nEpochs: 10,
            policyLearningRate: 1e-3,
            valueLearningRate: 1e-3,
            clipRatio: 0.2,
            targetKL: 0.01,
            useSDE: false,
            netArch: {
                'pi': [32, 32],
                'vf': [32, 32]
            },
            activation: 'relu',
            verbose: 0
        };
        this.config = Object.assign({}, configDefault, config);
        // Prepare network architecture
        if (Array.isArray(this.config.netArch)) {
            this.config.netArch = {
                'pi': this.config.netArch,
                'vf': this.config.netArch
            };
        }
        // Initialize logger
        this.log = (...args) => {
            if (this.config.verbose > 0) {
                console.log('[PPO]', ...args);
            }
        };
        // Initialize environment
        this.env = env;
        if ((this.env.actionSpace.class == 'Discrete') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'int32';
        }
        else if ((this.env.actionSpace.class == 'Box') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'float32';
        }
        // Initialize counters
        this.numTimesteps = 0;
        this.lastObservation = null;
        // Initialize buffer
        this.buffer = new Buffer(config);
        // Initialize models for actor and critic
        this.actor = this.createActor();
        this.critic = this.createCritic();
        // Initialize logStd (for continuous action space)
        if (this.env.actionSpace.class == 'Box') {
            this.logStd = tf.variable(tf.zeros([this.env.actionSpace.shape[0]]), true, 'logStd');
        }
        // Initialize optimizers
        this.optPolicy = tf.train.adam(this.config.policyLearningRate);
        this.optValue = tf.train.adam(this.config.valueLearningRate);
    }
    createActor() {
        const input = tf.layers.input({ shape: this.env.observationSpace.shape });
        let l = input;
        this.config.netArch?.pi.forEach((units) => {
            l = tf.layers.dense({
                units,
                activation: this.config.activation || 'relu'
            }).apply(l);
        });
        if (this.env.actionSpace.class === 'Discrete') {
            l = tf.layers.dense({
                units: this.env.actionSpace.n,
                activation: 'linear' // or another appropriate activation function
            }).apply(l);
        }
        else if (this.env.actionSpace.class === 'Box') {
            l = tf.layers.dense({
                units: this.env.actionSpace.shape[0],
                activation: 'linear' // or another appropriate activation function
            }).apply(l);
        }
        else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
        }
        return tf.model({ inputs: input, outputs: l });
    }
    createCritic() {
        const input = tf.layers.input({ shape: this.env.observationSpace.shape });
        let l = input;
        this.config.netArch?.vf.forEach((units) => {
            l = tf.layers.dense({
                units,
                activation: this.config.activation || 'relu'
            }).apply(l);
        });
        l = tf.layers.dense({
            units: 1,
            activation: 'linear' // Linear activation for the output layer
        }).apply(l);
        return tf.model({ inputs: input, outputs: l });
    }
    sampleAction(observationT) {
        return tf.tidy(() => {
            const preds = tf.squeeze(this.actor.predict(observationT), [0]);
            let action;
            if (this.env.actionSpace.class === 'Discrete') {
                action = tf.squeeze(tf.multinomial(preds, 1, this.randomSeed), [0]); // For discrete action space
            }
            else if (this.env.actionSpace.class === 'Box') {
                action = tf.add(tf.mul(tf.randomStandardNormal([this.env.actionSpace.shape[0]], 'float32', this.randomSeed), tf.exp(this.logStd)), preds); // For continuous action space
            }
            else {
                throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
            }
            return [preds, action];
        });
    }
    logProbCategorical(logits, x) {
        return tf.tidy(() => {
            const numActions = logits.shape[logits.shape.length - 1];
            const logprobabilitiesAll = tf.logSoftmax(logits);
            return tf.sum(tf.mul(tf.oneHot(x.toInt(), numActions), logprobabilitiesAll), logprobabilitiesAll.shape.length - 1);
        });
    }
    logProbNormal(loc, scale, x) {
        return tf.tidy(() => {
            const logUnnormalized = tf.mul(-0.5, tf.square(tf.sub(tf.div(x, scale), tf.div(loc, scale))));
            const logNormalization = tf.add(tf.scalar(0.5 * Math.log(2.0 * Math.PI)), tf.log(scale));
            return tf.sum(tf.sub(logUnnormalized, logNormalization), logUnnormalized.shape.length - 1);
        });
    }
    logProb(preds, actions) {
        if (this.env.actionSpace.class === 'Discrete') {
            return this.logProbCategorical(preds, actions);
        }
        else if (this.env.actionSpace.class === 'Box') {
            return this.logProbNormal(preds, tf.exp(this.logStd), actions);
        }
        else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
        }
    }
    predict(observation, deterministic = false) {
        if (observation instanceof Array) {
            observation = tf.tensor2d(observation, [1, observation.length]);
        }
        return this.actor.predict(observation);
    }
    predictAction(observation, deterministic = false) {
        const action = tf.tidy(() => {
            const pred = tf.squeeze(this.predict(observation, true), [0]);
            const actions = tf.squeeze(tf.multinomial(pred.arraySync(), 1), [0]);
            const action = actions.dataSync()[0];
            return action;
        });
        return action;
    }
    trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT) {
        const clipRatio = this.config.clipRatio ?? 0.2; // Provide a default value if undefined
        const optFunc = () => {
            const predsT = this.actor.predict(observationBufferT);
            const diffT = tf.sub(this.logProb(predsT, actionBufferT), logprobabilityBufferT);
            const ratioT = tf.exp(diffT);
            const minAdvantageT = tf.where(tf.greater(advantageBufferT, 0), tf.mul(tf.add(1, clipRatio), advantageBufferT), tf.mul(tf.sub(1, clipRatio), advantageBufferT));
            const policyLoss = tf.neg(tf.mean(tf.minimum(tf.mul(ratioT, advantageBufferT), minAdvantageT)));
            return policyLoss;
        };
        return tf.tidy(() => {
            const { value, grads } = this.optPolicy.computeGradients(optFunc);
            this.optPolicy.applyGradients(grads);
            const klTensor = tf.mean(tf.sub(logprobabilityBufferT, this.logProb(this.actor.predict(observationBufferT), actionBufferT)));
            // Ensure kl is a scalar by calling arraySync() on a scalar tensor
            const kl = klTensor.arraySync();
            return kl;
        });
    }
    trainValue(observationBufferT, returnBufferT) {
        const optFunc = () => {
            const valuesPredT = this.critic.predict(observationBufferT);
            return tf.losses.meanSquaredError(returnBufferT, valuesPredT);
        };
        tf.tidy(() => {
            const { value, grads } = this.optValue.computeGradients(optFunc);
            this.optValue.applyGradients(grads);
        });
    }
    _initCallback(callback) {
        if (typeof callback === 'function') {
            // Convert to 'unknown' first, then to the constructor type
            const callbackConstructor = callback;
            return new callbackConstructor();
        }
        if (typeof callback === 'object' && callback !== null) {
            return new DictCallback(callback);
        }
        return new BaseCallback();
    }
    async collectRollouts(callback) {
        if (this.lastObservation === null) {
            this.lastObservation = await this.env.reset();
        }
        this.buffer.reset();
        await callback.onRolloutStart(this);
        let sumReturn = 0;
        let sumLength = 0;
        let numEpisodes = 0;
        for (let step = 0; step < this.config.nSteps; step++) {
            // Predict action, value, and logprobability from last observation
            const [preds, action, value, logprobability] = tf.tidy(() => {
                const lastObservationT = tf.tensor([this.lastObservation]);
                const [predsT, actionT] = this.sampleAction(lastObservationT);
                const valueT = this.critic.predict(lastObservationT);
                const logprobabilityNum = this.logProb(predsT, actionT).dataSync()[0];
                const valueT_data = valueT.arraySync();
                return [
                    predsT.arraySync(),
                    actionT.arraySync(),
                    valueT_data,
                    logprobabilityNum
                ];
            });
            // Take action in environment
            const [newObservation, reward, done] = await this.env.step(action);
            sumReturn += reward;
            sumLength += 1;
            // Update global timestep counter
            this.numTimesteps += 1;
            await callback.onStep({ ppo: this, action, reward, done });
            this.buffer.add(this.lastObservation, action, reward, value, logprobability);
            this.lastObservation = newObservation;
            if (done || step === this.config.nSteps - 1) {
                const lastValue = done
                    ? 0
                    : tf.tidy(() => {
                        const prediction = this.critic.predict(tf.tensor([newObservation]));
                        return prediction.arraySync()[0][0];
                    });
                this.buffer.finishTrajectory(lastValue);
                numEpisodes += 1;
                this.lastObservation = await this.env.reset();
            }
        }
        await callback.onRolloutEnd(this);
    }
    async train() {
        // Get values from the buffer
        const [observationBuffer, actionBuffer, advantageBuffer, returnBuffer, logprobabilityBuffer,] = this.buffer.get();
        const actionBufferShape = Array.isArray(actionBuffer[0]) ? [actionBuffer.length, actionBuffer[0].length] : [actionBuffer.length];
        const [observationBufferT, actionBufferT, advantageBufferT, returnBufferT, logprobabilityBufferT] = tf.tidy(() => [
            tf.tensor(observationBuffer),
            tf.tensor(actionBuffer, actionBufferShape),
            tf.tensor(advantageBuffer),
            tf.tensor(returnBuffer).reshape([-1, 1]),
            tf.tensor(logprobabilityBuffer)
        ]);
        for (let i = 0; i < this.config.nEpochs; i++) {
            const kl = this.trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT);
            if (kl > 1.5 * this.config.targetKL) {
                break;
            }
        }
        for (let i = 0; i < this.config.nEpochs; i++) {
            this.trainValue(observationBufferT, returnBufferT);
        }
        tf.dispose([
            observationBufferT,
            actionBufferT,
            advantageBufferT,
            returnBufferT,
            logprobabilityBufferT
        ]);
    }
    async learn(learnConfig) {
        const learnConfigDefault = {
            'totalTimesteps': 1000,
            'logInterval': 1,
            'callback': null
        };
        let { totalTimesteps, logInterval, callback } = { ...learnConfigDefault, ...learnConfig };
        callback = this._initCallback(callback);
        let iteration = 0;
        await callback.onTrainingStart(this);
        while (this.numTimesteps < totalTimesteps) {
            await this.collectRollouts(callback);
            iteration += 1;
            if (logInterval && iteration % logInterval === 0) {
                this.log(`Timesteps: ${this.numTimesteps}`);
            }
            await this.train();
        }
        await callback.onTrainingEnd(this);
    }
    _serialize(object = this) {
        const attributes = {};
        for (const key in object) {
            if (object.hasOwnProperty(key) && typeof object[key] !== 'function' && !(object[key] instanceof tf.LayersModel)) {
                attributes[key] = object[key];
            }
        }
        return attributes;
    }
    _deserialize(data, object = this) {
        for (const key in data) {
            if (data.hasOwnProperty(key) && object.hasOwnProperty(key)) {
                object[key] = data[key];
            }
        }
    }
    async _convertModelWeightsToJSON(model) {
        const weights = model.getWeights();
        const weightData = weights.map(w => w.arraySync());
        return weightData;
    }
    async toJSON() {
        //Get PPO Attributes
        const model_object = this._serialize();
        model_object['buffer'] = this._serialize(this.buffer);
        const [actor_w, critic_w] = await Promise.all([
            this._convertModelWeightsToJSON(this.actor),
            this._convertModelWeightsToJSON(this.critic)
        ]);
        model_object['actor'] = {
            'architecture': this.actor.toJSON(null, false),
            'weights': actor_w
        };
        model_object['critic'] = {
            'architecture': this.critic.toJSON(null, false),
            'weights': critic_w
        };
        const model_json = JSON.stringify(model_object);
        return model_json;
    }
    async fromJSON(jsonString) {
        const modelObject = JSON.parse(jsonString);
        // Rebuild the PPO Config & Buffer
        this.config = modelObject.config;
        this._deserialize(modelObject.buffer, this.buffer);
        // Rebuild the actor and critic models
        this.actor = await this._rebuildModelFromJSON(modelObject.actor);
        this.critic = await this._rebuildModelFromJSON(modelObject.critic);
    }
    async _rebuildModelFromJSON(modelData) {
        // Create a model from the architecture
        const model = await tf.loadLayersModel(tf.io.fromMemory(modelData.architecture));
        // Restore the weights
        const weights = modelData.weights.map((w) => tf.tensor(w));
        model.setWeights(weights);
        return model;
    }
    _checkPackageSave(path) {
        //Check if Running in Node - and Throw an Error If Not
        if (typeof window !== 'undefined') {
            throw new Error('This method only works in node');
        }
        //Check if Path Exists - and Throw an Error If Not
        if (!fs.existsSync(path)) {
            throw new Error('Path does not exist');
        }
        return true;
    }
    async savePackage(path, callback) {
        this._checkPackageSave(path);
        //Save the Actor and Critic Models
        fs.mkdirSync(`${path}/actor`, { recursive: true });
        fs.mkdirSync(`${path}/critic`, { recursive: true });
        //Save the PPO Config & Buffer
        const model_object = this._serialize();
        model_object['buffer'] = this._serialize(this.buffer);
        const model_json = JSON.stringify(model_object);
        const saved_models = Promise.all([
            fs.writeFile(`${path}/model.json`, model_json, 'utf-8', () => { }),
            this.actor.save(`file://${path}/actor`),
            this.critic.save(`file://${path}/critic`),
        ]);
        if (callback) {
            await saved_models.catch((err) => { throw new Error(err); }).finally(() => callback());
        }
        else {
            await saved_models.catch((err) => { throw new Error(err); });
        }
    }
    async loadPackage(path, callback) {
        this._checkPackageSave(path);
        //Load the Actor and Critic Models
        const model_json = fs.readFileSync(`${path}/model.json`, 'utf-8');
        const model_object = JSON.parse(model_json);
        const [actor, critic] = await Promise.all([
            tf.loadLayersModel(`file://${path}/actor/model.json`),
            tf.loadLayersModel(`file://${path}/critic/model.json`),
        ]);
        // Rebuild the PPO Config & Buffer
        this.config = model_object.config;
        this._deserialize(model_object.buffer, this.buffer);
        // Rebuild the actor and critic models
        this.actor = actor;
        this.critic = critic;
        if (callback) {
            callback();
        }
    }
    setRandomSeed(seed) {
        this.randomSeed = seed;
    }
}
if (typeof module === 'object' && module.exports) {
    module.exports = PPO;
}
