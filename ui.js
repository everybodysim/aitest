import { MultiAgentTrainer } from './training.js';

export function createUI({ trainer, agents, onResetAll, onSpawnObject, onToggleFollow }) {
  const panel = document.createElement('div');
  panel.id = 'ui-panel';

  const goalOptions = MultiAgentTrainer.goalOptions()
    .map((g) => `<option value="${g.key}">${g.label}</option>`)
    .join('');

  panel.innerHTML = `
    <h3 style="margin:0 0 10px 0;">Humanoid ML Sandbox</h3>
    <div class="row">
      <button id="resetBtn">Reset</button>
      <button id="trainToggle">Pause Training</button>
    </div>
    <div class="row">
      <select id="goalSelect">${goalOptions}</select>
      <button id="followBtn">Follow Agent 1</button>
    </div>
    <div class="row">
      <select id="spawnType">
        <option value="box">Spawn Box</option>
        <option value="sphere">Spawn Sphere</option>
        <option value="ramp">Spawn Ramp</option>
      </select>
      <button id="spawnBtn">Spawn</button>
    </div>
    <div class="row">
      <button id="saveBtn">Save Model</button>
      <button id="loadBtn">Load Model</button>
    </div>
    <div id="learnSummary" style="font-size:12px;margin:8px 0 4px 0;color:#dbeafe"></div>
    <canvas id="learnGraph" width="292" height="92" style="width:100%;height:92px;background:#0b1220;border:1px solid #334155;border-radius:6px;margin-bottom:8px"></canvas>
    <div id="stats"></div>
    <div id="hint">Editor mode drag: hold Shift and drag dynamic objects. Water region applies reduced gravity + drag.</div>
  `;

  document.body.appendChild(panel);

  const trainToggle = panel.querySelector('#trainToggle');
  panel.querySelector('#resetBtn').onclick = onResetAll;
  panel.querySelector('#goalSelect').onchange = (e) => trainer.setGoal(e.target.value);

  trainToggle.onclick = () => {
    trainer.toggleTraining(!trainer.enabled);
    trainToggle.textContent = trainer.enabled ? 'Pause Training' : 'Resume Training';
  };

  panel.querySelector('#followBtn').onclick = () => onToggleFollow();
  panel.querySelector('#spawnBtn').onclick = () => {
    const kind = panel.querySelector('#spawnType').value;
    onSpawnObject(kind);
  };

  panel.querySelector('#saveBtn').onclick = () => trainer.save();
  panel.querySelector('#loadBtn').onclick = async () => {
    const ok = await trainer.load();
    if (!ok) alert('No saved model found in localStorage.');
  };

  const summaryEl = panel.querySelector('#learnSummary');
  const graph = panel.querySelector('#learnGraph');
  const gctx = graph.getContext('2d');
  const statsEl = panel.querySelector('#stats');

  function drawGraph(history) {
    const w = graph.width;
    const h = graph.height;
    gctx.clearRect(0, 0, w, h);

    gctx.strokeStyle = '#1e293b';
    gctx.lineWidth = 1;
    for (let i = 1; i <= 3; i += 1) {
      const y = (h / 4) * i;
      gctx.beginPath();
      gctx.moveTo(0, y);
      gctx.lineTo(w, y);
      gctx.stroke();
    }

    if (history.length < 2) return;

    const minV = Math.min(...history);
    const maxV = Math.max(...history);
    const range = Math.max(1e-4, maxV - minV);

    gctx.strokeStyle = '#22d3ee';
    gctx.lineWidth = 2;
    gctx.beginPath();
    history.forEach((v, i) => {
      const x = (i / (history.length - 1)) * (w - 1);
      const y = h - 5 - ((v - minV) / range) * (h - 10);
      if (i === 0) gctx.moveTo(x, y);
      else gctx.lineTo(x, y);
    });
    gctx.stroke();
  }

  function renderStats() {
    const summary = trainer.getTrainingSummary();
    const trendArrow = summary.trend >= 0 ? '↗' : '↘';
    summaryEl.innerHTML = `Batches: ${summary.batchCount} &nbsp; Last mean reward: ${summary.lastBatchMeanReward.toFixed(3)} &nbsp; Trend: ${trendArrow} ${summary.trend.toFixed(3)} &nbsp; Explore: ${summary.explorationStd.toFixed(2)}`;
    drawGraph(summary.history);

    statsEl.innerHTML = agents
      .map(
        (agent) => `
      <div class="agent-row">
        <b>Agent ${agent.id + 1}</b><br>
        Reward: ${agent.stats.reward.toFixed(2)}<br>
        Distance: ${agent.stats.distance.toFixed(2)} m<br>
        Time alive: ${agent.stats.timeAlive.toFixed(1)} s<br>
        Finesse: ${agent.getFinesse().toFixed(1)}
      </div>
    `,
      )
      .join('');
  }

  return { renderStats };
}
