import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

/** ----------------------------
 * Callback interfaces
 * -----------------------------*/
export interface CallbackObject {
  onTrainingStart?: (ctx: PPO) => Promise<void> | void;
  onTrainingEnd?: (ctx: PPO) => Promise<void> | void;
  onRolloutStart?: (ctx: PPO) => Promise<void> | void;
  onRolloutEnd?: (ctx: PPO) => Promise<void> | void;
  onStep?: (info: { ppo: PPO; action: number; reward: number; done: boolean }) => Promise<void> | void;
}

export class BaseCallback {
  async onTrainingStart(_ctx: PPO) {}
  async onTrainingEnd(_ctx: PPO) {}
  async onRolloutStart(_ctx: PPO) {}
  async onRolloutEnd(_ctx: PPO) {}
  async onStep(_info: { ppo: PPO; action: number; reward: number; done: boolean }) {}
}

export class DictCallback extends BaseCallback {
  dict: CallbackObject;
  constructor(dict: CallbackObject) {
    super();
    this.dict = dict;
  }
  async onTrainingStart(ctx: PPO) { if (this.dict.onTrainingStart) await this.dict.onTrainingStart(ctx); }
  async onTrainingEnd(ctx: PPO) { if (this.dict.onTrainingEnd) await this.dict.onTrainingEnd(ctx); }
  async onRolloutStart(ctx: PPO) { if (this.dict.onRolloutStart) await this.dict.onRolloutStart(ctx); }
  async onRolloutEnd(ctx: PPO) { if (this.dict.onRolloutEnd) await this.dict.onRolloutEnd(ctx); }
  async onStep(info: { ppo: PPO; action: number; reward: number; done: boolean }) {
    if (this.dict.onStep) await this.dict.onStep(info);
  }
}

/** ----------------------------
 * Buffer (GAE-Lambda)
 * -----------------------------*/
class Buffer {
  gamma: number;
  lam: number;
  observationBuffer: number[][] = [];
  actionBuffer: number[] = [];
  rewardBuffer: number[] = [];
  valueBuffer: number[] = [];
  logprobabilityBuffer: number[] = [];
  advantageBuffer: number[] = [];
  returnBuffer: number[] = [];
  trajectoryStartIndex = 0;
  pointer = 0;

  constructor(cfg: { gamma: number; lam: number }) {
    this.gamma = cfg.gamma;
    this.lam = cfg.lam;
  }

  reset(): void {
    this.observationBuffer = [];
    this.actionBuffer = [];
    this.rewardBuffer = [];
    this.valueBuffer = [];
    this.logprobabilityBuffer = [];
    this.advantageBuffer = [];
    this.returnBuffer = [];
    this.trajectoryStartIndex = 0;
    this.pointer = 0;
  }

  add(obs: number[], action: number, reward: number, value: number, logp: number) {
    this.observationBuffer.push(obs);
    this.actionBuffer.push(action);
    this.rewardBuffer.push(reward);
    this.valueBuffer.push(value);
    this.logprobabilityBuffer.push(logp);
    this.pointer += 1;
  }

  private discountedCumulativeSums(arr: number[], coeff: number): number[] {
    const res: number[] = [];
    let s = 0;
    for (let i = arr.length - 1; i >= 0; i--) {
      s = arr[i] + coeff * s;
      res[i] = s;
    }
    return res;
  }

  finishTrajectory(lastValue: number): void {
    const rewards = this.rewardBuffer.slice(this.trajectoryStartIndex, this.pointer).concat(lastValue * this.gamma);
    const values = this.valueBuffer.slice(this.trajectoryStartIndex, this.pointer).concat(lastValue);
    const deltas = rewards.slice(0, -1).map((r, i) => r + this.gamma * values[i + 1] - values[i]);
    const advantages = this.discountedCumulativeSums(deltas, this.gamma * this.lam);
    const returns = this.discountedCumulativeSums(rewards, this.gamma).slice(0, -1);
    this.advantageBuffer = this.advantageBuffer.concat(advantages);
    this.returnBuffer = this.returnBuffer.concat(returns);
    this.trajectoryStartIndex = this.pointer;
  }

  get(): [number[][], number[], number[], number[], number[]] {
    // Normalize advantages with plain JS
    const adv = this.advantageBuffer;
    let sum = 0; for (let i = 0; i < adv.length; i++) sum += adv[i];
    const mean = adv.length ? sum / adv.length : 0;
    let vsum = 0; for (let i = 0; i < adv.length; i++) { const d = adv[i] - mean; vsum += d * d; }
    const std = adv.length ? Math.sqrt(vsum / adv.length) || 1 : 1;
    this.advantageBuffer = adv.map(a => (a - mean) / std);
    return [this.observationBuffer, this.actionBuffer, this.advantageBuffer, this.returnBuffer, this.logprobabilityBuffer];
  }
}

/** ----------------------------
 * PPO
 * -----------------------------*/

export interface PPOConfig {
  nSteps?: number;
  nEpochs?: number;
  policyLearningRate?: number;
  valueLearningRate?: number;
  clipRatio?: number;
  targetKL?: number;
  useSDE?: boolean;
  netArch?: { pi?: number[]; vf?: number[] } | number[];
  activation?: any;
  verbose?: number;
  /** rollout collection mode */
  mode?: 'sequential' | 'vectorized';
  /** number of parallel environments when mode='vectorized' */
  nEnvs?: number;
  /** factory to create new env instances for vectorized mode */
  makeEnv?: () => any | Promise<any>;
}

export default class PPO {
  env: any;
  config: PPOConfig;
  actor: tf.LayersModel;
  critic: tf.LayersModel;
  logStd: tf.Variable | null = null; // for continuous
  optPolicy: tf.Optimizer;
  optValue: tf.Optimizer;
  randomSeed = 0;
  lastObservation: number[] | null = null;
  numTimesteps = 0;

  buffer: Buffer;

  constructor(env: any, config: PPOConfig = {}) {
    const cfg: PPOConfig = {
      nSteps: 512,
      nEpochs: 10,
      policyLearningRate: 1e-3,
      valueLearningRate: 1e-3,
      clipRatio: 0.2,
      targetKL: 0.01,
      useSDE: false,
      netArch: { pi: [32, 32], vf: [32, 32] },
      activation: 'relu',
      verbose: 0,
      mode: 'sequential',
      nEnvs: 1,
      ...config,
    };
    this.config = cfg;
    if (Array.isArray(cfg.netArch)) {
      this.config.netArch = { pi: cfg.netArch, vf: cfg.netArch };
    }

    this.env = env;
    this.actor = this.createActor();
    this.critic = this.createCritic();

    if (this.env.actionSpace.class === 'Box') {
      this.logStd = tf.variable(tf.zeros([this.env.actionSpace.shape[0]]), true, 'logStd');
    }

    this.optPolicy = tf.train.adam(this.config.policyLearningRate!);
    this.optValue = tf.train.adam(this.config.valueLearningRate!);

    this.buffer = new Buffer({ gamma: 0.99, lam: 0.95 });
  }

  private denseStack(input: tf.SymbolicTensor, units: number[]): tf.SymbolicTensor {
    let x: tf.SymbolicTensor = input;
    for (const u of units) {
      x = tf.layers.dense({ units: u, activation: this.config.activation || 'relu' }).apply(x) as tf.SymbolicTensor;
    }
    return x;
  }

  createActor(): tf.LayersModel {
    const input = tf.layers.input({ shape: this.env.observationSpace.shape });
    let x = this.denseStack(input, (this.config.netArch as any).pi);
    if (this.env.actionSpace.class === 'Discrete') {
      x = tf.layers.dense({ units: this.env.actionSpace.n, activation: 'linear' }).apply(x) as tf.SymbolicTensor;
    } else {
      x = tf.layers.dense({ units: this.env.actionSpace.shape[0], activation: 'linear' }).apply(x) as tf.SymbolicTensor;
    }
    return tf.model({ inputs: input, outputs: x });
  }

  createCritic(): tf.LayersModel {
    const input = tf.layers.input({ shape: this.env.observationSpace.shape });
    const x = this.denseStack(input, (this.config.netArch as any).vf);
    const v = tf.layers.dense({ units: 1, activation: 'linear' }).apply(x) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: v });
  }

  predict(obs: tf.Tensor | number[]): tf.Tensor {
    const x = Array.isArray(obs) ? tf.tensor2d([obs as number[]]) : obs;
    return this.actor.predict(x) as tf.Tensor;
  }

  chooseMostLikelyResponse<T extends number[] | tf.Tensor>(logProbs: T): number {
    if (Array.isArray(logProbs)) {
      let bestI = 0; let bestV = -Infinity;
      for (let i = 0; i < logProbs.length; i++) { const v = logProbs[i]; if (v > bestV) { bestV = v; bestI = i; } }
      return bestI;
    }
    return tf.tidy(() => (logProbs as tf.Tensor).argMax().dataSync()[0]);
  }

  sampleAction(observationT: tf.Tensor): [tf.Tensor, tf.Tensor] {
    return tf.tidy(() => {
      const logits = tf.squeeze(this.actor.predict(observationT) as tf.Tensor, [0]); // [A] or [D]
      let action: tf.Tensor;
      if (this.env.actionSpace.class === 'Discrete') {
        const logits2D = (logits as tf.Tensor1D).expandDims(0) as tf.Tensor2D;
        action = tf.squeeze(tf.multinomial(logits2D, 1, this.randomSeed), [0]); // []
      } else if (this.env.actionSpace.class === 'Box') {
        action = tf.add(
          tf.mul(tf.randomStandardNormal([this.env.actionSpace.shape[0]], 'float32', this.randomSeed), tf.exp(this.logStd!)),
          logits
        );
      } else {
        throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
      }
      return [logits, action];
    });
  }

  logProbCategorical(logits: tf.Tensor, x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const numActions = logits.shape[logits.shape.length - 1];
      const lsm = tf.logSoftmax(logits);
      const oneHot = tf.oneHot(x as tf.Tensor1D, numActions);
      return tf.sum(tf.mul(oneHot, lsm), -1);
    });
  }

  logProbNormal(mu: tf.Tensor, std: tf.Tensor, x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const varT = tf.square(std);
      const logScale = tf.log(std).sum(-1);
      const diff = tf.sub(x, mu);
      const quadratic = tf.sum(tf.square(diff).div(varT), -1).mul(0.5);
      const constTerm = (mu.shape[mu.shape.length - 1]) * 0.5 * Math.log(2 * Math.PI);
      return tf.mul(-1, quadratic.add(logScale).add(constTerm));
    });
  }

  logProb(preds: tf.Tensor, actions: tf.Tensor): tf.Tensor {
    if (this.env.actionSpace.class === 'Discrete') {
      return this.logProbCategorical(preds, actions);
    } else if (this.env.actionSpace.class === 'Box') {
      return this.logProbNormal(preds, tf.exp(this.logStd!), actions);
    }
    throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
  }

  predictAction(observation: tf.Tensor | number[], deterministic = false): number {
    if (deterministic) {
      return tf.tidy(() => {
        const pred = tf.squeeze(this.predict(observation), [0]);
        return pred.argMax().dataSync()[0];
      });
    }
    return tf.tidy(() => {
      const pred1D = tf.squeeze(this.predict(observation), [0]);
      const logits2D = (pred1D as tf.Tensor1D).expandDims(0) as tf.Tensor2D;
      const sampled = tf.squeeze(tf.multinomial(logits2D, 1, this.randomSeed), [0]);
      return sampled.dataSync()[0];
    });
  }

  predictProbabilities(observation: tf.Tensor | number[]): number[] {
    return tf.tidy(() => {
      const logits = tf.squeeze(this.predict(observation), [0]);
      const probs = tf.softmax(logits);
      return probs.arraySync() as number[];
    });
  }

  async collectRollouts(callback: BaseCallback): Promise<void> {
    if (this.lastObservation === null) this.lastObservation = await this.env.reset();
    this.buffer.reset();
    await callback.onRolloutStart(this);

    for (let step = 0; step < (this.config.nSteps || 0); step++) {
      const [action, value, logprobability] = tf.tidy(() => {
        const lastObservationT = tf.tensor2d([this.lastObservation as number[]]);
        const [predsT, actionT] = this.sampleAction(lastObservationT);
        const valueT = this.critic.predict(lastObservationT) as tf.Tensor;
        const logProbT = this.logProb(predsT, actionT) as tf.Tensor1D;
        const a = (actionT as tf.Tensor).dataSync()[0];
        const v = (valueT as tf.Tensor).dataSync()[0];
        const lp = logProbT.dataSync()[0];
        return [a, v, lp];
      });

      const [newObservation, reward, done] = await this.env.step(action);
      await callback.onStep({ ppo: this, action, reward, done });

      this.buffer.add(this.lastObservation as number[], action, reward, value, logprobability);
      this.lastObservation = newObservation;

      if (done || step === (this.config.nSteps || 0) - 1) {
        const lastValue = done ? 0 : tf.tidy(() => {
          const newObservationT = tf.tensor2d([newObservation]);
          const prediction = this.critic.predict(newObservationT) as tf.Tensor2D;
          const result = (prediction as tf.Tensor).dataSync()[0] as number;
          newObservationT.dispose();
          prediction.dispose();
          return result;
        });
        this.buffer.finishTrajectory(lastValue as number);
        this.lastObservation = await this.env.reset();
      }
    }

    await callback.onRolloutEnd(this);
  }

  async collectRolloutsVectorized(callback: BaseCallback, nEnvs: number): Promise<void> {
    // Function to spawn/clone envs without reprocessing heavy data.
    const spawnEnv = async (i: number) => {
      // Preferred: explicit factory if available
      if (this.config.makeEnv) {
        return await Promise.resolve(this.config.makeEnv());
      }
      // Try common cloning methods on the provided env
      const base: any = this.env;
      for (const fn of ['clone', 'copy', 'fork', 'spawn']) {
        if (typeof base[fn] === 'function') {
          const inst = await Promise.resolve(base[fn](i));
          if (inst) return inst;
        }
      }
      // As a last resort, try to construct a new instance from constructor with a shared context if provided
      try {
        if (base && base.constructor) {
          if (typeof base.getContext === 'function') {
            return new (base.constructor as any)(base.getContext());
          } else {
            return new (base.constructor as any)();
          }
        }
      } catch {}
      throw new Error('[PPO] vectorized mode requires either config.makeEnv() or an env.clone()/copy()/fork()/spawn() method (optionally using shared, preprocessed data).');
    };

    // Create N envs
    const envs: any[] = [];
    for (let i = 0; i < nEnvs; i++) {
      const e = await spawnEnv(i);
      envs.push(e);
    }

    // Reset all
    let obs: any[] = await Promise.all(envs.map(e => e.reset()));
    const dones: boolean[] = new Array(nEnvs).fill(false);
    const buffers = Array.from({ length: nEnvs }, () => new Buffer({ gamma: 0.99, lam: 0.95 }));

    await callback.onRolloutStart(this);

    for (let t = 0; t < (this.config.nSteps || 0); t++) {
      const [actions, values, logps] = tf.tidy(() => {
        const obsT = tf.tensor2d(obs);     // [N, obs_dim]
        const logits = this.actor.predict(obsT) as tf.Tensor;    // [N, A] or [N, D]
        const valuesT = this.critic.predict(obsT) as tf.Tensor;  // [N, 1]
        let actionsT: tf.Tensor;
        if (this.env.actionSpace.class === 'Discrete') {
          actionsT = tf.squeeze(tf.multinomial(logits as tf.Tensor2D, 1, this.randomSeed), [1]); // [N]
        } else if (this.env.actionSpace.class === 'Box') {
          const std = tf.exp(this.logStd!);          // [D]
          const D = (logits as tf.Tensor2D).shape[1];
          const noise = tf.randomStandardNormal([nEnvs, D], 'float32', this.randomSeed);
          actionsT = tf.add(tf.mul(noise, std), logits); // [N, D]
        } else {
          throw new Error('Unknown action space class: ' + this.env.actionSpace.class);
        }
        const logpT = this.logProb(logits, actionsT as tf.Tensor);
        const a = Array.from((actionsT as tf.Tensor).dataSync());
        const v = Array.from((valuesT as tf.Tensor).dataSync());
        const lp = Array.from((logpT as tf.Tensor).dataSync());
        return [a, v, lp];
      });

      const stepResults = await Promise.all(envs.map((e, i) => e.step((actions as any)[i])));
      for (let i = 0; i < nEnvs; i++) {
        const [nextObs, reward, done] = stepResults[i];
        const value = Array.isArray((values as any)[i]) ? (values as any)[i][0] : (values as any)[i];
        buffers[i].add(obs[i], (actions as any)[i], reward, value as number, (logps as any)[i] as number);
        await callback.onStep({ ppo: this, action: (actions as any)[i], reward, done });
        obs[i] = nextObs;
        dones[i] = !!done;
        if (dones[i]) {
          buffers[i].finishTrajectory(0);
          obs[i] = await envs[i].reset();
          dones[i] = false;
        }
      }
    }

    // Horizon reached; bootstrap for unfinished episodes
    for (let i = 0; i < nEnvs; i++) {
      const lastValue = tf.tidy(() => {
        const o = tf.tensor2d([obs[i]]);
        const v = this.critic.predict(o) as tf.Tensor2D;
        const out = (v as tf.Tensor).dataSync()[0] as number;
        o.dispose(); (v as tf.Tensor).dispose();
        return out;
      });
      buffers[i].finishTrajectory(lastValue);
    }

    // Merge buffers into the main buffer
    this.buffer.reset();
    for (const b of buffers) {
      const [ob, ac, ad, rets, lp] = b.get();
      this.buffer.observationBuffer.push(...ob);
      this.buffer.actionBuffer.push(...ac);
      this.buffer.advantageBuffer.push(...ad);
      this.buffer.returnBuffer.push(...rets);
      this.buffer.logprobabilityBuffer.push(...lp);
      this.buffer.pointer += ob.length;
    }

    await callback.onRolloutEnd(this);
  }

  trainPolicy(obsT: tf.Tensor, actT: tf.Tensor, oldLogpT: tf.Tensor, advT: tf.Tensor): number {
    const clipRatio = this.config.clipRatio ?? 0.2;
    const optFunc = () => {
      const logits = this.actor.predict(obsT) as tf.Tensor;
      const logp = this.logProb(logits, actT);
      const ratio = tf.exp(tf.sub(logp, oldLogpT));
      const clipped = tf.clipByValue(ratio, 1 - clipRatio, 1 + clipRatio).mul(advT);
      const loss = tf.neg(tf.minimum(ratio.mul(advT), clipped).mean());
      return loss as tf.Scalar;
    };
    const loss = this.optPolicy.minimize(optFunc, true) as tf.Scalar | null;
    const kl = tf.tidy(() => {
      const logits = this.actor.predict(obsT) as tf.Tensor;
      const logp = this.logProb(logits, actT);
      const ratio = tf.exp(tf.sub(logp, oldLogpT));
      const klT = tf.mean(tf.sub(ratio, tf.log(ratio)).sub(1).mul(-1));
      const out = (klT as tf.Tensor).dataSync()[0];
      return out;
    });
    if (loss) loss.dispose();
    return kl;
  }

  trainValue(obsT: tf.Tensor, retT: tf.Tensor): void {
    const optFunc = () => {
      const values = this.critic.predict(obsT) as tf.Tensor;
      return tf.mean(tf.losses.meanSquaredError(retT, values)) as tf.Scalar;
    };
    tf.tidy(() => {
      const { grads } = this.optValue.computeGradients(optFunc);
      this.optValue.applyGradients(grads);
    });
  }

  async train(): Promise<void> {
    const [obs, act, adv, ret, logp] = this.buffer.get();
    const obsT = tf.tensor2d(obs);
    const actT = tf.tensor1d(act, 'int32');
    const advT = tf.tensor1d(adv);
    const retT = tf.tensor1d(ret);
    const logpT = tf.tensor1d(logp);

    const targetKL = this.config.targetKL ?? 0.01;

    for (let epoch = 0; epoch < (this.config.nEpochs || 0); epoch++) {
      const kl = this.trainPolicy(obsT, actT, logpT, advT);
      if (kl > 1.5 * targetKL) break;
      this.trainValue(obsT, retT);
    }

    obsT.dispose(); actT.dispose(); advT.dispose(); retT.dispose(); logpT.dispose();
  }

  _initCallback(cb?: CallbackObject | BaseCallback | null): BaseCallback {
    if (!cb) return new BaseCallback();
    if (cb instanceof BaseCallback) return cb;
    return new DictCallback(cb as CallbackObject);
  }

  async learn(cfg: { totalTimesteps?: number; logInterval?: number; callback?: CallbackObject | BaseCallback; mode?: 'sequential' | 'vectorized'; nEnvs?: number } = {}) {
    const totalTimesteps = cfg.totalTimesteps ?? 2048;
    const logInterval = cfg.logInterval ?? 1;
    const callback = this._initCallback(cfg.callback ?? null);
    const mode = cfg.mode ?? this.config.mode ?? 'sequential';
    const nEnvs = cfg.nEnvs ?? this.config.nEnvs ?? 1;

    await callback.onTrainingStart(this);
    let timesteps = 0;
    let it = 0;

    while (timesteps < totalTimesteps) {
      if (mode === 'vectorized' && nEnvs > 1) {
        await this.collectRolloutsVectorized(callback, nEnvs);
        timesteps += (this.config.nSteps || 0) * nEnvs;
      } else {
        await this.collectRollouts(callback);
        timesteps += (this.config.nSteps || 0);
      }
      await this.train();
      it += 1;
      if (it % logInterval === 0 && this.config.verbose) {
        console.log(`[PPO] Iteration ${it} timesteps=${timesteps} mode=${mode} nEnvs=${nEnvs}`);
      }
    }
    await callback.onTrainingEnd(this);
  }

  setRandomSeed(seed: number) { this.randomSeed = seed; }

  /** --------- Minimal JSON persistence for demo (weights only) ---------- */
  async save(path: string) {
    const actSave = await this.actor.save(tf.io.withSaveHandler(async d => {
      fs.writeFileSync(path + '.actor.json', JSON.stringify(d));
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' as const } };
    }));
    const crtSave = await this.critic.save(tf.io.withSaveHandler(async d => {
      fs.writeFileSync(path + '.critic.json', JSON.stringify(d));
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' as const } };
    }));
    return { actSave, crtSave };
  }

  async load(path: string) {
    const actorData = JSON.parse(fs.readFileSync(path + '.actor.json', 'utf8'));
    const criticData = JSON.parse(fs.readFileSync(path + '.critic.json', 'utf8'));
    // @ts-ignore
    this.actor = await tf.loadLayersModel(tf.io.fromMemory(actorData.modelTopology, actorData.weightSpecs, actorData.weightData));
    // @ts-ignore
    this.critic = await tf.loadLayersModel(tf.io.fromMemory(criticData.modelTopology, criticData.weightSpecs, criticData.weightData));
  }
}
