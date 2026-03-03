const tf = window.tf;

if (!tf) {
  throw new Error('TensorFlow.js failed to load. Check blocked CDN policy and network filters.');
}

const GOALS = {
  walk_forward: {
    name: 'Walk Forward',
    reward(agent) {
      const torso = agent.torsoBody;
      const q = torso.quaternion;
      const upY = 1 - 2 * (q.x * q.x + q.z * q.z);
      const upBonus = Math.max(0, upY);
      return torso.velocity.x * 1.1 + upBonus * 0.2;
    },
  },
  reach_target: {
    name: 'Reach Target',
    target: { x: 28, z: 0 },
    reward(agent) {
      const torso = agent.torsoBody;
      const dx = this.target.x - torso.position.x;
      const dz = this.target.z - torso.position.z;
      const d = Math.hypot(dx, dz);
      return 1.2 / (1 + d) + torso.velocity.x * 0.3;
    },
  },
  stay_upright: {
    name: 'Stay Upright',
    reward(agent) {
      const q = agent.torsoBody.quaternion;
      const upY = 1 - 2 * (q.x * q.x + q.z * q.z);
      return upY * 1.0 - Math.abs(agent.torsoBody.angularVelocity.x) * 0.06;
    },
  },
  jump_plateau: {
    name: 'Jump Onto Plateau',
    reward(agent) {
      const t = agent.torsoBody;
      const targetX = 12;
      const targetY = 3.2;
      const toward = 1 / (1 + Math.abs(targetX - t.position.x));
      const height = Math.max(0, t.position.y - targetY) * 0.45;
      return toward + height;
    },
  },
};

function postureAndIdlePenalty(agent) {
  const torso = agent.torsoBody;
  const q = torso.quaternion;
  const upY = 1 - 2 * (q.x * q.x + q.z * q.z);
  const speed = Math.hypot(torso.velocity.x, torso.velocity.z);
  const lowPosturePenalty = Math.max(0, 0.65 - upY) * 0.9;
  const idlePenalty = speed < 0.08 ? 0.03 : 0;
  return lowPosturePenalty + idlePenalty;
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

    this.obsSize = agents[0].observationSize;
    this.actSize = agents[0].actionSize;

    this.model = tf.sequential({
      layers: [
        tf.layers.dense({ units: 128, activation: 'relu', inputShape: [this.obsSize] }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: this.actSize, activation: 'tanh' }),
      ],
    });
    this.optimizer = tf.train.adam(0.0008);

    this.buffer = [];
  }

  setGoal(goalKey) {
    this.goal = GOALS[goalKey] || GOALS.walk_forward;
  }

  static goalOptions() {
    return Object.entries(GOALS).map(([key, g]) => ({ key, label: g.name }));
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
      return out.arraySync();
    });

    this.agents.forEach((a, idx) => a.applyActions(actions[idx]));

    if (!this.enabled) return;

    for (let i = 0; i < this.agents.length; i += 1) {
      const baseReward = this.goal.reward(this.agents[i]);
      const penalty = postureAndIdlePenalty(this.agents[i]);
      const reward = baseReward - penalty;
      this.agents[i].stats.reward += reward;
      this.buffer.push({ obs: observations[i], act: actions[i], reward });
    }

    this.stepCount += 1;
    if (this.stepCount % this.batchSteps === 0) {
      this.lastBatchMeanReward = this.buffer.reduce((acc, b) => acc + b.reward, 0) / Math.max(1, this.buffer.length);
      this.trainingHistory.push(this.lastBatchMeanReward);
      if (this.trainingHistory.length > 180) this.trainingHistory.shift();
      this.batchCount += 1;

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
