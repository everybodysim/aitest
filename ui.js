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

  const statsEl = panel.querySelector('#stats');

  function renderStats() {
    statsEl.innerHTML = agents
      .map(
        (agent) => `
      <div class="agent-row">
        <b>Agent ${agent.id + 1}</b><br>
        Reward: ${agent.stats.reward.toFixed(2)}<br>
        Distance: ${agent.stats.distance.toFixed(2)} m<br>
        Time alive: ${agent.stats.timeAlive.toFixed(1)} s
      </div>
    `,
      )
      .join('');
  }

  return { renderStats };
}
