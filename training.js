const tf = window.tf;

if (!tf) {
  throw new Error('TensorFlow.js failed to load. Check blocked CDN policy and network filters.');
}

const GOALS = {
  walk_forward: {
    name: 'Walk Forward',
    reward(agent) {
      const torso = agent.torsoBody;
      const forwardSpeed = Math.max(0, torso.velocity.x);
      const distanceFromStart = Math.max(0, torso.position.x - agent.start.x);
      return forwardSpeed * 1.5 + distanceFromStart * 0.09;
    },
  },
};

function erraticMotionPenalty(agent) {
  const bodyList = Object.values(agent.bodies);
  const avgAngVel = bodyList.reduce((acc, b) => acc + b.angularVelocity.length(), 0) / bodyList.length;
  return Math.max(0, avgAngVel - 5.0) * 0.05;
}

export class MultiAgentTrainer {
  constructor(agents) {
    this.agents = agents;
    this.goal = GOALS.walk_forward;
    this.enabled = true;
    this.gamma = 0.98;
    this.batchSteps = 96;
    this.stepCount = 0;
    this.batchCount = 0;
    this.lastBatchMeanReward = 0;
    this.trainingHistory = [];
    this.explorationStd = 0.35;
    this.minExplorationStd = 0.06;

    this.obsSize = agents[0].observationSize;
    this.actSize = agents[0].actionSize;

    this.model = tf.sequential({
      layers: [
        tf.layers.dense({ units: 128, activation: 'relu', inputShape: [this.obsSize] }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: this.actSize, activation: 'tanh' }),
      ],
    });
    this.optimizer = tf.train.adam(0.0015);

    this.buffer = [];
  }

  setGoal() {
    this.goal = GOALS.walk_forward;
  }

  static goalOptions() {
    return [{ key: 'walk_forward', label: GOALS.walk_forward.name }];
  }

  toggleTraining(value) {
    this.enabled = value;
  }

  getTrainingSummary() {
    const head = this.trainingHistory[this.trainingHistory.length - 1] || 0;
    const tail = this.trainingHistory[Math.max(0, this.trainingHistory.length - 25)] || 0;
    return {
      batchCount: this.batchCount,
      lastBatchMeanReward: this.lastBatchMeanReward,
      trend: head - tail,
      history: this.trainingHistory,
      explorationStd: this.explorationStd,
    };
  }

  async save() {
    await this.model.save('localstorage://humanoid-policy');
  }

  async load() {
    try {
      this.model = await tf.loadLayersModel('localstorage://humanoid-policy');
      return true;
    } catch {
      return false;
    }
  }

  step() {
    const observations = this.agents.map((a) => a.getObservations());
    const actions = tf.tidy(() => {
      const obsTensor = tf.tensor2d(observations);
      const out = this.model.predict(obsTensor);
      const means = out.arraySync();
      return means.map((row) => row.map((v) => {
        if (!this.enabled) return v;
        const noisy = v + (Math.random() * 2 - 1) * this.explorationStd;
        return Math.max(-1, Math.min(1, noisy));
      }));
    });

    this.agents.forEach((a, idx) => a.applyActions(actions[idx]));

    if (!this.enabled) return;

    for (let i = 0; i < this.agents.length; i += 1) {
      const baseReward = this.goal.reward(this.agents[i]);
      const penalty = erraticMotionPenalty(this.agents[i]);
      const reward = baseReward - penalty;
      this.agents[i].stats.reward += reward;
      this.agents[i].lifetimeRewardSum += reward;
      this.agents[i].lifetimeRewardCount += 1;
      this.buffer.push({ obs: observations[i], act: actions[i], reward });
    }

    this.stepCount += 1;
    if (this.stepCount % this.batchSteps === 0) {
      this.lastBatchMeanReward = this.buffer.reduce((acc, b) => acc + b.reward, 0) / Math.max(1, this.buffer.length);
      this.trainingHistory.push(this.lastBatchMeanReward);
      if (this.trainingHistory.length > 180) this.trainingHistory.shift();
      this.batchCount += 1;
      this.explorationStd = Math.max(this.minExplorationStd, this.explorationStd * 0.996);

      this.trainBatch();
      this.buffer.length = 0;
    }
  }

  trainBatch() {
    if (this.buffer.length < 40) return;

    const returns = new Array(this.buffer.length);
    let running = 0;
    for (let i = this.buffer.length - 1; i >= 0; i -= 1) {
      running = this.buffer[i].reward + this.gamma * running;
      returns[i] = running;
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdev = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length) + 1e-6;
    const adv = returns.map((v) => (v - mean) / stdev);

    const xs = tf.tensor2d(this.buffer.map((b) => b.obs));
    const ys = tf.tensor2d(this.buffer.map((b) => b.act));
    const advTensor = tf.tensor2d(adv, [adv.length, 1]);

    this.optimizer.minimize(() => {
      const pred = this.model.apply(xs);
      const logProb = tf.neg(tf.square(tf.sub(pred, ys)).mean(1, true));
      const weighted = tf.mul(logProb, advTensor).mean();
      return tf.neg(weighted);
    });

    tf.dispose([xs, ys, advTensor]);
  }
}
