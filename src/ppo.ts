import * as tf from '@tensorflow/tfjs';

function log(...args: any[]) {
    console.log('[PPO]', ...args);
}

interface LearnConfig {
    totalTimesteps?: number;
    logInterval?: number;
    callback: any
}

type CallbackObject = {
    onRolloutEnd: (p: PPO) => void;
    onRolloutStart: (p: PPO) => void;
    onTrainingEnd: (p: PPO) => void;
    onTrainingStart: (p: PPO) => void;
    onStep: (p: PPO) => boolean;
}

class BaseCallback {
    nCalls: number;
    constructor() {
        this.nCalls = 0
    }

    _onStep(alg: any) { return true }
    onStep(alg: any) {
        this.nCalls += 1
        return this._onStep(alg)
    }

    _onTrainingStart(alg: any) {}
    onTrainingStart(alg: any) {
        this._onTrainingStart(alg)
    }

    _onTrainingEnd(alg: any) {}
    onTrainingEnd(alg: any) {
        this._onTrainingEnd(alg)
    }

    _onRolloutStart(alg: any) {}
    onRolloutStart(alg: any) {
        this._onRolloutStart(alg)
    }

    _onRolloutEnd(alg: any) {}
    onRolloutEnd(alg: any) {
        this._onRolloutEnd(alg)
    }
}

class FunctionalCallback extends BaseCallback {
    callback: any;
    constructor(callback: any) {
        super()
        this.callback = callback
    }

    override _onStep(alg: this) {
        if (this.callback) {
            return this.callback(alg)
        }
        return true
    }
}

class DictCallback extends BaseCallback {
    callback: CallbackObject;
    constructor(callback: CallbackObject) {
        super()
        this.callback = callback
    }

    override _onStep(alg: any) {
        if (this.callback && this.callback.onStep) {
            return this.callback.onStep(alg)
        }
        return true
    }
    
    override _onTrainingStart(alg: any) {
        if (this.callback && this.callback.onTrainingStart) {
            this.callback.onTrainingStart(alg)
        }
    }

    override _onTrainingEnd(alg: any) {
        if (this.callback && this.callback.onTrainingEnd) {
            this.callback.onTrainingEnd(alg)
        }
    }

    override _onRolloutStart(alg: any) {
        if (this.callback && this.callback.onRolloutStart) {
            this.callback.onRolloutStart(alg)
        }
    }

    override _onRolloutEnd(alg: any) {
        if (this.callback && this.callback.onRolloutEnd) {
            this.callback.onRolloutEnd(alg)
        }
    }
}

class Buffer {
    bufferConfig: {
        gamma: number;
        lam: number;
    };
    gamma: number;
    lam: number;
    observationBuffer: any[] = [];
    actionBuffer: any[]  = []; 
    rewardBuffer: number[]  = [];
    valueBuffer: number[]  = [];
    logprobabilityBuffer: number[]  = [];
    advantageBuffer: number[]  = [];
    returnBuffer: number[]  = [];
    trajectoryStartIndex: number = 0;
    pointer: number = 0;

    constructor(bufferConfig: { gamma?: number; lam?: number; [key: string]: any }) {
        const bufferConfigDefault = {
            gamma: 0.99,
            lam: 0.95
        };
        this.bufferConfig = { ...bufferConfigDefault, ...bufferConfig };
        this.gamma = this.bufferConfig.gamma;
        this.lam = this.bufferConfig.lam;
        this.reset();
    }

    add(observation: any, action: any, reward: number, value: number, logprobability: number): void {
        this.observationBuffer.push(observation);
        this.actionBuffer.push(action);
        this.rewardBuffer.push(reward);
        this.valueBuffer.push(value);
        this.logprobabilityBuffer.push(logprobability);
        this.pointer += 1;
    }

    discountedCumulativeSums(arr: number[], coeff: number): number[] {
        let res: number[] = [];
        let s = 0;
        arr.reverse().forEach(v => {
            s = v + s * coeff;
            res.push(s);
        });
        return res.reverse();
    }

    finishTrajectory(lastValue: number): void {
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

    get(): [any[], any[], number[], number[], number[]] {
        let advantageMean: number, advantageStd: number;
        [advantageMean, advantageStd] = tf.tidy(() => [
            tf.mean(this.advantageBuffer).arraySync() as number,
            tf.moments(this.advantageBuffer).variance.sqrt().arraySync() as number
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

    reset(): void {
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

interface PPOConfig {
    nSteps?: number;
    nEpochs?: number;
    policyLearningRate?: number;
    valueLearningRate?: number;
    clipRatio?: number;
    targetKL?: number;
    useSDE?: boolean;
    netArch?: {
        pi?: number[];
        vf?: number[];
    };
    activation?: any;
    verbose?: number;
}

export class PPO {
    config: PPOConfig;
    env: any;
    numTimesteps: number;
    lastObservation: any;
    buffer: Buffer;
    actor: any;
    critic: any;
    logStd: any;
    optPolicy: any;
    optValue: any;
    log: (...args: any[]) => void;

    constructor(env: any, config: PPOConfig) {
        const configDefault: PPOConfig = {
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
            }
        }

        // Initialize logger
        this.log = (...args) => {
            if (this.config.verbose! > 0) {
                console.log('[PPO]', ...args)
            }
        }

        // Initialize environment
        this.env = env
        if ((this.env.actionSpace.class == 'Discrete') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'int32'
        } else if ((this.env.actionSpace.class == 'Box') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'float32'
        }

        // Initialize counters
        this.numTimesteps = 0
        this.lastObservation = null

        // Initialize buffer
        this.buffer = new Buffer(config)

        // Initialize models for actor and critic
        this.actor = this.createActor()
        this.critic = this.createCritic()

        // Initialize logStd (for continuous action space)
        if (this.env.actionSpace.class == 'Box') {
            this.logStd = tf.variable(tf.zeros([this.env.actionSpace.shape[0]]), true, 'logStd')
        }

        // Initialize optimizers
        this.optPolicy = tf.train.adam(this.config.policyLearningRate)
        this.optValue = tf.train.adam(this.config.valueLearningRate)
    }

    createActor(): tf.LayersModel {
        const input = tf.layers.input({ shape: this.env.observationSpace.shape });
        let l: tf.SymbolicTensor = input;

        this.config.netArch?.pi!.forEach((units: number) => {
            l = tf.layers.dense({
                units,
                activation: this.config.activation || 'relu'
            }).apply(l) as tf.SymbolicTensor;
        });

        if (this.env.actionSpace.class === 'Discrete') {
            l = tf.layers.dense({
                units: this.env.actionSpace.n,
                activation: 'linear' // or another appropriate activation function
            }).apply(l) as tf.SymbolicTensor;
        } else if (this.env.actionSpace.class === 'Box') {
            l = tf.layers.dense({
                units: this.env.actionSpace.shape[0],
                activation: 'linear' // or another appropriate activation function
            }).apply(l) as tf.SymbolicTensor;
        } else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
        }

        return tf.model({ inputs: input, outputs: l });
    }

    createCritic(): tf.LayersModel {
        const input = tf.layers.input({ shape: this.env.observationSpace.shape });
        let l: tf.SymbolicTensor = input;

        this.config.netArch?.vf!.forEach((units: number) => {
            l = tf.layers.dense({
                units,
                activation: this.config.activation || 'relu'
            }).apply(l) as tf.SymbolicTensor;
        });

        l = tf.layers.dense({
            units: 1,
            activation: 'linear' // Linear activation for the output layer
        }).apply(l) as tf.SymbolicTensor;

        return tf.model({ inputs: input, outputs: l });
    }

    sampleAction(observationT: tf.Tensor): [tf.Tensor, tf.Tensor] {
        return tf.tidy(() => {
            const preds = tf.squeeze(this.actor.predict(observationT), [0]);
            let action: tf.Tensor;

            if (this.env.actionSpace.class === 'Discrete') {
                action = tf.squeeze(tf.multinomial(preds as tf.Tensor2D, 1), [0]); // For discrete action space
            } else if (this.env.actionSpace.class === 'Box') {
                action = tf.add(
                    tf.mul(
                        tf.randomStandardNormal([this.env.actionSpace.shape[0]]), 
                        tf.exp(this.logStd)
                    ),
                    preds
                ); // For continuous action space
            } else {
                throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
            }

            return [preds, action];
        });
    }

    logProbCategorical(logits: tf.Tensor, x: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            const numActions = logits.shape[logits.shape.length - 1];
            const logprobabilitiesAll = tf.logSoftmax(logits);
            return tf.sum(
                tf.mul(tf.oneHot(x.toInt(), numActions), logprobabilitiesAll),
                logprobabilitiesAll.shape.length - 1
            );
        });
    }

    logProbNormal(loc: tf.Tensor, scale: tf.Tensor, x: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            const logUnnormalized = tf.mul(
                -0.5,
                tf.square(
                    tf.sub(
                        tf.div(x, scale),
                        tf.div(loc, scale)
                    )
                )
            );
            const logNormalization = tf.add(
                tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
                tf.log(scale)
            );
            return tf.sum(
                tf.sub(logUnnormalized, logNormalization),
                logUnnormalized.shape.length - 1
            );
        });
    }

    logProb(preds: tf.Tensor, actions: tf.Tensor): tf.Tensor {
        if (this.env.actionSpace.class === 'Discrete') {
            return this.logProbCategorical(preds, actions);
        } else if (this.env.actionSpace.class === 'Box') {
            return this.logProbNormal(preds, tf.exp(this.logStd), actions);
        } else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
        }
    }
    
    predict(observation: tf.Tensor, deterministic: boolean = false): tf.Tensor {
        return this.actor.predict(observation);
    }

    trainPolicy(
        observationBufferT: tf.Tensor, 
        actionBufferT: tf.Tensor, 
        logprobabilityBufferT: tf.Tensor, 
        advantageBufferT: tf.Tensor
    ): number {
        const clipRatio = this.config.clipRatio ?? 0.2; // Provide a default value if undefined
    
        const optFunc = () => {
            const predsT = this.actor.predict(observationBufferT) as tf.Tensor;
            const diffT = tf.sub(
                this.logProb(predsT, actionBufferT),
                logprobabilityBufferT
            );
            const ratioT = tf.exp(diffT);
            const minAdvantageT = tf.where(
                tf.greater(advantageBufferT, 0),
                tf.mul(tf.add(1, clipRatio), advantageBufferT),
                tf.mul(tf.sub(1, clipRatio), advantageBufferT)
            );
            const policyLoss = tf.neg(tf.mean(
                tf.minimum(tf.mul(ratioT, advantageBufferT), minAdvantageT)
            ));
            return policyLoss;
        };
    
        return tf.tidy(() => {
            const { value, grads } = this.optPolicy.computeGradients(optFunc);
            this.optPolicy.applyGradients(grads);
            const klTensor = tf.mean(tf.sub(
                logprobabilityBufferT,
                this.logProb(this.actor.predict(observationBufferT) as tf.Tensor, actionBufferT)
            ));
            // Ensure kl is a scalar by calling arraySync() on a scalar tensor
            const kl = klTensor.arraySync() as number;
            return kl;
        });
    }

    trainValue(observationBufferT: tf.Tensor, returnBufferT: tf.Tensor): void {
        const optFunc = () => {
            const valuesPredT = this.critic.predict(observationBufferT) as tf.Tensor;
            return tf.losses.meanSquaredError(returnBufferT, valuesPredT);
        };

        tf.tidy(() => {
            const { value, grads } = this.optValue.computeGradients(optFunc);
            this.optValue.applyGradients(grads);
        });
    }

    _initCallback(callback: CallbackObject | BaseCallback | null): BaseCallback {
        if (typeof callback === 'function') {
            // Convert to 'unknown' first, then to the constructor type
            const callbackConstructor = callback as unknown as new () => BaseCallback;
            return new callbackConstructor();
        }
        if (typeof callback === 'object' && callback !== null) {
            return new DictCallback(callback as CallbackObject);
        }
        return new BaseCallback();
    }
    

    async collectRollouts(callback: BaseCallback): Promise<void> {
        if (this.lastObservation === null) {
            this.lastObservation = await this.env.reset();
        }
    
        this.buffer.reset();
        callback.onRolloutStart(this);
    
        let sumReturn = 0;
        let sumLength = 0;
        let numEpisodes = 0;
    
        for (let step = 0; step < this.config.nSteps!; step++) {
            // Predict action, value, and logprobability from last observation
                const [preds, action, value, logprobability] = tf.tidy(() => {
                const lastObservationT = tf.tensor([this.lastObservation]);
                const [predsT, actionT] = this.sampleAction(lastObservationT);
                const valueT = this.critic.predict(lastObservationT) as tf.Tensor;
                const logprobabilityT = this.logProb(predsT, actionT);
                const logprobabilityNum: number = Array.isArray(logprobabilityT) ? logprobabilityT[0] : logprobabilityT;
                const valueT_data: any = valueT.arraySync();

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
    
            callback.onStep(this);
    
            this.buffer.add(
                this.lastObservation,
                action,
                reward,
                value,
                logprobability
            );
    
            this.lastObservation = newObservation;
    
            if (done || step === this.config.nSteps! - 1){
                const lastValue = done 
                    ? 0 
                    : tf.tidy(() => this.critic.predict(tf.tensor([newObservation])).arraySync())[0][0];
                this.buffer.finishTrajectory(lastValue);
                numEpisodes += 1;
                this.lastObservation = await this.env.reset();
            }
        }
    
        callback.onRolloutEnd(this);
    }

    async train() {
        // Get values from the buffer
        const [
            observationBuffer,
            actionBuffer,
            advantageBuffer,
            returnBuffer,
            logprobabilityBuffer,
        ] = this.buffer.get();
    
        const actionBufferShape = Array.isArray(actionBuffer[0]) ? [actionBuffer.length, actionBuffer[0].length] : [actionBuffer.length];
        console.log(actionBufferShape);

        const [
            observationBufferT,
            actionBufferT,
            advantageBufferT,
            returnBufferT,
            logprobabilityBufferT
        ] = tf.tidy(() => [
                tf.tensor(observationBuffer),
                tf.tensor(actionBuffer, actionBufferShape),
                tf.tensor(advantageBuffer),
                tf.tensor(returnBuffer).reshape([-1, 1]),
                tf.tensor(logprobabilityBuffer)
            ]
        );
    
        for (let i = 0; i < this.config.nEpochs!; i++) {
            const kl = this.trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT);
            if (kl > 1.5 * this.config.targetKL!) {
                break;
            }
        }
    
        for (let i = 0;  i < this.config.nEpochs!; i++) {
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

    async learn(learnConfig: LearnConfig) {
        const learnConfigDefault = {
            'totalTimesteps': 1000,
            'logInterval': 1,
            'callback': null
        };
        let { 
            totalTimesteps,
            logInterval,
            callback
        } = { ...learnConfigDefault, ...learnConfig };
    
        callback = this._initCallback(callback);
        
        let iteration = 0;
        
        callback.onTrainingStart(this);
    
        while (this.numTimesteps < totalTimesteps) {
            await this.collectRollouts(callback);
            iteration += 1;
            if (logInterval && iteration % logInterval === 0) {
                this.log(`Timesteps: ${this.numTimesteps}`);
            }
            await this.train();
        }
        
        callback.onTrainingEnd(this);
    }
}

//if (typeof module === 'object' && module.exports) {
    module.exports = PPO
//}