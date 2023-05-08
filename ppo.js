// Check if node
// if (typeof window === 'undefined') {
if (typeof module === 'object' && module.exports) {
    var tf = require('@tensorflow/tfjs-node-gpu')
}

function log () {
    // console.log('[PPO]', ...arguments)
}

class BaseCallback {
    constructor() {
        this.nCalls = 0
    }

    _onStep(alg) { return true }
    onStep(alg) {
        this.nCalls += 1
        return this._onStep(alg)
    }

    _onTrainingStart(alg) {}
    onTrainingStart(alg) {
        this._onTrainingStart(alg)
    }

    _onTrainingEnd(alg) {}
    onTrainingEnd(alg) {
        this._onTrainingEnd(alg)
    }

    _onRolloutStart(alg) {}
    onRolloutStart(alg) {
        this._onRolloutStart(alg)
    }

    _onRolloutEnd(alg) {}
    onRolloutEnd(alg) {
        this._onRolloutEnd(alg)
    }
}

class FunctionalCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback) {
            return this.callback(alg)
        }
        return true
    }
}

class DictCallback extends BaseCallback {
    constructor(callback) {
        super()
        this.callback = callback
    }

    _onStep(alg) {
        if (this.callback && this.callback.onStep) {
            return this.callback.onStep(alg)
        }
        return true
    }
    
    _onTrainingStart(alg) {
        if (this.callback && this.callback.onTrainingStart) {
            this.callback.onTrainingStart(alg)
        }
    }

    _onTrainingEnd(alg) {
        if (this.callback && this.callback.onTrainingEnd) {
            this.callback.onTrainingEnd(alg)
        }
    }

    _onRolloutStart(alg) {
        if (this.callback && this.callback.onRolloutStart) {
            this.callback.onRolloutStart(alg)
        }
    }

    _onRolloutEnd(alg) {
        if (this.callback && this.callback.onRolloutEnd) {
            this.callback.onRolloutEnd(alg)
        }
    }
}






const bufferConfigDefault = {
    gamma: 0.99,
    lam: 0.95
}

class Buffer {
    constructor(bufferConfig) {
        this.bufferConfig = Object.assign({}, bufferConfigDefault, bufferConfig)
        this.gamma = this.bufferConfig.gamma
        this.lam = this.bufferConfig.lam
        this.reset()
    }

    add(observation, action, reward, value, logprobability) {
        // log('!', action)
        this.observationBuffer.push(observation.slice(0))
        this.actionBuffer.push(action)
        this.rewardBuffer.push(reward)
        this.valueBuffer.push(value)
        this.logprobabilityBuffer.push(logprobability)
        this.pointer += 1
    }

    discountedCumulativeSums (arr, coeff) {
        var res = []
        var s = 0
        arr.reverse().forEach(v => {
            s = v + s * coeff
            res.push(s)
        })
        return res.reverse()
    }

    finishTrajectory(lastValue) {
        //log('>', this.rewardBuffer, this.rewardBuffer.length)
        var rewards = this.rewardBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue * this.gamma)
        //log('>', rewards, rewards.length)
        var values = this.valueBuffer
            .slice(this.trajectoryStartIndex, this.pointer)
            .concat(lastValue)
        //log('>', values, values.length)
        var deltas = rewards
            .slice(0, -1)
            .map((reward, ri) => reward - (values[ri] - this.gamma * values[ri + 1]))
        //log('deltas>', deltas.length)
        
        this.advantageBuffer = this.advantageBuffer
            .concat(this.discountedCumulativeSums(deltas, this.gamma * this.lam))
        this.returnBuffer = this.returnBuffer
            .concat(this.discountedCumulativeSums(rewards, this.gamma).slice(0, -1))
                
        this.trajectoryStartIndex = this.pointer

        // log('>', this.observationBuffer.length)
        // log('!', this.actionBuffer)
        // log('>', this.advantageBuffer.length)
        // log('>', this.returnBuffer.length)
        // log('>', this.rewardBuffer.length)
        // log('>', deltas.length)

        // process.exit(0)
    }

    get() {
        const [advantageMean, advantageStd] = tf.tidy(() => [
            tf.mean(this.advantageBuffer).arraySync(),
            tf.moments(this.advantageBuffer).variance.sqrt().arraySync()
        ])
        
        this.advantageBuffer = this.advantageBuffer
            .map(advantage => (advantage - advantageMean) / advantageStd)
        
        return [
            this.observationBuffer,
            this.actionBuffer,
            this.advantageBuffer,
            this.returnBuffer,
            this.logprobabilityBuffer
        ]
    }

    reset() {
        this.observationBuffer = []
        this.actionBuffer = []
        this.advantageBuffer = []
        this.rewardBuffer = []
        this.returnBuffer = []
        this.valueBuffer = []
        this.logprobabilityBuffer = []
        this.trajectoryStartIndex = 0
        this.pointer = 0
    }

}

/*
class ActorCriticPolicy {
    constructor(policyConfig) {
        const policyConfigDefault = {
            netArch: {
                'pi': [16, 16],
                'vf': [16, 16]
            },
            activationFn: 'tanh',
            shareFeaturesExtractor: true,
        }
        this.policyConfig = Object.assign({}, policyConfigDefault, policyConfig)
        this.observationSpace = this.policyConfig.observationSpace
        this.actionSpace = this.policyConfig.actionSpace
        this.netArch = this.policyConfig.netArch
        this.activationFn = this.policyConfig.activationFn
        this.shareFeaturesExtractor = this.policyConfig.shareFeaturesExtractor
    }
    
    _makeFeaturesExtractor() {
    }
}
*/

class PPO {
    constructor(env, config) {
        const configDefault = {
            nSteps: 512,
            nEpochs: 10,
            policyLearningRate: 1e-3,
            valueLearningRate: 1e-3,
            clipRatio: 0.2,
            targetKL: 0.01,
            useSDE: false, // TODO: State Dependent Exploration (gSDE)
        }

        this.config = Object.assign({}, configDefault, config)
        this.env = env
        if ((this.env.actionSpace.class == 'Discrete') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'int32'
        } else if ((this.env.actionSpace.class == 'Box') && !this.env.actionSpace.dtype) {
            this.env.actionSpace.dtype = 'float32'
        }
        this.numTimesteps = 0
        this.lastObservation = null

        var input = tf.layers.input({shape: this.env.observationSpace.shape})
        var l = tf.layers.dense({units: 32, activation: 'relu'}).apply(input)
        l = tf.layers.dense({units: 32, activation: 'relu'}).apply(l)
        if (this.env.actionSpace.class == 'Discrete') {
            l = tf.layers.dense({
                units: this.env.actionSpace.n, 
                kernelInitializer: 'glorotNormal'
            }).apply(l)
        } else if (this.env.actionSpace.class == 'Box') {
            l = tf.layers.dense({
                units: this.env.actionSpace.shape[0], 
                kernelInitializer: 'glorotNormal'
            }).apply(l)
        } else {
            throw new Error('Unknown action space class: ' + this.env.actionSpace.class)
        }
        this.actor = tf.model({inputs: input, outputs: l})

        var input = tf.layers.input({shape: this.env.observationSpace.shape})
        var l = tf.layers.dense({units: 32, activation: 'relu'}).apply(input)
        l = tf.layers.dense({units: 32, activation: 'relu'}).apply(l)
        l = tf.layers.dense({units: 1, activation: 'linear', kernelInitializer: 'glorotNormal'}).apply(l)
        this.critic = tf.model({inputs: input, outputs: l})

        if (this.env.actionSpace.class == 'Box') {
            this.logStd = tf.variable(tf.zeros([this.env.actionSpace.shape[0]]), true, 'logStd')
        }

        this.buffer = new Buffer(config)

        this.optPolicy = tf.train.adam(this.config.policyLearningRate)
        this.optValue = tf.train.adam(this.config.valueLearningRate)
    }

    sampleAction(observationT) {
        return tf.tidy(() => {
            const preds = tf.squeeze(this.actor.predict(observationT), 0)
            let action 
            if (this.env.actionSpace.class == 'Discrete') {
                action = tf.squeeze(tf.multinomial(preds, 1), 0) // > []
            } else if (this.env.actionSpace.class == 'Box') {
                action = tf.add(
                    tf.mul(
                        tf.randomStandardNormal([this.env.actionSpace.shape[0]]), 
                        tf.exp(this.logStd)
                    ),
                    preds
                ) // > [actionSpace.shape[0]]
            }
            return [preds, action]
        })
    }

    logProbCategorical(logits, x) {
        return tf.tidy(() => {
            const numActions = logits.shape[logits.shape.length - 1]
            const logprobabilitiesAll = tf.logSoftmax(logits)
            return tf.sum(
                tf.mul(tf.oneHot(x, numActions), logprobabilitiesAll),
                logprobabilitiesAll.shape.length - 1
            )
        })
    }
    
    logProbNormal(loc, scale, x) {
        return tf.tidy(() => {
            const logUnnormalized = tf.mul(
                -0.5,
                tf.square(
                    tf.sub(
                        tf.div(x, scale),
                        tf.div(loc, scale)
                    )
                )
            )
            const logNormalization = tf.add(
                tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
                tf.log(scale)
            )
            return tf.sum(
                tf.sub(logUnnormalized, logNormalization),
                logUnnormalized.shape.length - 1
            )
        })
    }

    logProb(preds, actions) {
        // Preds can be logits or means
        if (this.env.actionSpace.class == 'Discrete') {
            return this.logProbCategorical(preds, actions)
        } else if (this.env.actionSpace.class == 'Box') {
            return this.logProbNormal(preds, tf.exp(this.logStd), actions)
        }
    }
    
    predict(observation, deterministic=false) {
        return this.actor.predict(observation)
    }

    trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT) {
        var optFunc = () => {
            const predsT = this.actor.predict(observationBufferT) // -> Logits or means
            const diffT = tf.sub(
                this.logProb(predsT, actionBufferT),
                logprobabilityBufferT
            )
            const ratioT = tf.exp(diffT)
            const minAdvantageT = tf.where(
                tf.greater(advantageBufferT, 0),
                tf.mul(tf.add(1, this.config.clipRatio), advantageBufferT),
                tf.mul(tf.sub(1, this.config.clipRatio), advantageBufferT)
            )
            const policyLoss = tf.neg(tf.mean(
                tf.minimum(tf.mul(ratioT, advantageBufferT), minAdvantageT)
            ))
            // log('[Train Policy] preds:', preds.arraySync(), preds.shape)
            // log('[Train Policy] actionBuffer:', actionBuffer)
            // log('[Train Policy] logProb:', logProb.arraySync())
            // log('[Train Policy] diff:', diff.arraySync())
            // log('[Train Policy] ratio:', ratio.arraySync())
            // log('[Train Policy] logprobabilityBuffer:', logprobabilityBuffer)
            // log('[Train Policy] advantageBuffer:', advantageBuffer)
            // log('[Train Policy] policyLoss:', policyLoss.arraySync())
            return policyLoss
        }
    
        return tf.tidy(() => {
            const {values, grads} = this.optPolicy.computeGradients(optFunc)
            // log('logStd grads:', grads.logStd.arraySync())
            // log('denseKernel grads (avg):', tf.mean(grads['dense_Dense1/kernel']).arraySync())
            this.optPolicy.applyGradients(grads)
            const kl = tf.mean(tf.sub(
                logprobabilityBufferT,
                this.logProb(this.actor.predict(observationBufferT), actionBufferT)
            ))
            return kl.arraySync()
        })
    }

    trainValue(observationBufferT, returnBufferT) {
        var optFunc = () => {
            const valuesPredT = this.critic.predict(observationBufferT)
            return tf.losses.meanSquaredError(returnBufferT, valuesPredT)
        }
                
        tf.tidy(() => {
            var {values, grads} = this.optValue.computeGradients(optFunc)
            this.optValue.applyGradients(grads)
        })
    }

    _initCallback(callback) {
        // Function, not class
        if (typeof callback === 'function') {
            if (callback.prototype.constructor === undefined) {
                return new FunctionalCallback(callback)
            }
            return callback
        }
        if (typeof callback === 'object') {
            return new DictCallback(callback)
        }
        return new BaseCallback() 
        // TODO:List
        // TODO:Class
    }

    async collectRollouts(callback) {
        if (this.lastObservation === null) {
            this.lastObservation = this.env.reset()
        }

        this.buffer.reset()
        callback.onRolloutStart(this)

        var sumReturn = 0
        var sumLength = 0
        var numEpisodes = 0

        var allPreds = []
        var allActions = []
        var allClippedActions = []

        for (let step = 0; step < this.config.nSteps; step++) {
            // Predict action, value and logprob from last observation
            const [preds, action, value, logprobability] = tf.tidy(() => {
                const lastObservationT = tf.tensor([this.lastObservation])
                const [predsT, actionT] = this.sampleAction(lastObservationT)
                const valueT = this.critic.predict(lastObservationT)
                const logprobabilityT = this.logProb(predsT, actionT)
                return [
                    predsT.arraySync(), // -> Discrete: [actionSpace.n] or Box: [actionSpace.shape[0]]
                    actionT.arraySync(), // -> Discrete: [] or Box: [actionSpace.shape[0]]
                    valueT.arraySync()[0][0],
                    logprobabilityT.arraySync()
                ]
            })
            allPreds.push(preds)
            allActions.push(action)

            // Rescale for continuous action space
            let clippedAction = action
            if (this.env.actionSpace.class == 'Box') {
                let h = this.env.actionSpace.high
                let l = this.env.actionSpace.low
                if (typeof h === 'number' && typeof l === 'number') {
                    clippedAction = action.map(a => {
                        return Math.min(Math.max(a, l), h)
                    })
                }
            }
            allClippedActions.push(clippedAction)

            // Take action in environment
            const [newObservation, reward, done] = await this.env.step(clippedAction)
            sumReturn += reward
            sumLength += 1

            // Update global timestep counter
            this.numTimesteps += 1 

            callback.onStep(this)

            this.buffer.add(
                this.lastObservation, 
                action, 
                reward, 
                value, 
                logprobability
            )
            
            this.lastObservation = newObservation
            
            if (done || step === this.config.nSteps - 1) {
                const lastValue = done 
                    ? 0 
                    : tf.tidy(() => this.critic.predict(tf.tensor([newObservation])).arraySync())[0][0]
                this.buffer.finishTrajectory(lastValue)
                numEpisodes += 1
                this.lastObservation = this.env.reset()
            }
        }
            
        // log(`Timesteps: ${this.numTimesteps}, Episodes: ${numEpisodes}`)
        // log(`Avg returns: ${sumReturn / numEpisodes}`)
        // log(`Avg length: ${sumLength / numEpisodes}`)
        // log('All preds (avg):', tf.mean(tf.tensor(allPreds), 0).arraySync())
        // log('All actions (avg):', tf.mean(tf.tensor(allActions), 0).arraySync())
        // log('All clipped actions (avg):', tf.mean(tf.tensor(allClippedActions), 0).arraySync())

        callback.onRolloutEnd(this)
    }

    async train(config) {
        // Get values from the buffer
        const [
            observationBuffer,
            actionBuffer,
            advantageBuffer,
            returnBuffer,
            logprobabilityBuffer,
        ] = this.buffer.get()

        const [
            observationBufferT,
            actionBufferT,
            advantageBufferT,
            returnBufferT,
            logprobabilityBufferT
        ] = tf.tidy(() => [
            tf.tensor(observationBuffer),
            tf.tensor(actionBuffer, null, this.env.actionSpace.dtype),
            tf.tensor(advantageBuffer),
            tf.tensor(returnBuffer).reshape([-1, 1]),
            tf.tensor(logprobabilityBuffer)
        ])

        for (let i = 0; i < this.config.nEpochs; i++) {
            const kl = this.trainPolicy(observationBufferT, actionBufferT, logprobabilityBufferT, advantageBufferT)
            if (kl > 1.5 * this.config.targetKL) {
                break
            }
        }

        log('Train value net...')
        for (let i = 0;  i < this.config.nEpochs; i++) {
            this.trainValue(observationBufferT, returnBufferT)
        }

        tf.dispose([
            observationBufferT, 
            actionBufferT,
            advantageBufferT,
            returnBufferT,
            logprobabilityBufferT
        ])
    }

    async learn(learnConfig) {
        const learnConfigDefault = {
            'totalTimesteps': 1000,
            'logInterval': 1,
            'callback': null
        }
        let { 
            totalTimesteps,
            logInterval,
            callback
        } = Object.assign({}, learnConfigDefault, learnConfig)

        callback = this._initCallback(callback)
        
        let iteration = 0
        
        callback.onTrainingStart(this)

        log('Start')

        while (this.numTimesteps < totalTimesteps) {
            await this.collectRollouts(callback)
            iteration += 1
            if (logInterval && iteration % logInterval === 0) {
                log(`Timesteps: ${this.numTimesteps}`)
            }
            this.train()
        }
        
        callback.onTrainingEnd(this)
    }
}

if (typeof module === 'object' && module.exports) {
    module.exports = PPO
}