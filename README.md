# Humanoid Physics ML Sandbox

Browser-based training sandbox where five ragdoll humanoids learn torque-driven movement goals in real time.

## Stack

- Three.js for rendering
- cannon-es for physics
- TensorFlow.js for policy-gradient training

## Run

Because modules are loaded from local files + CDNs, serve the folder with any static server:

```bash
python -m http.server 8080
```

Then open `http://localhost:8080`.

## Controls

- **Reset**: reset all agents.
- **Goal dropdown**: switch reward target.
- **Pause/Resume Training**: toggle learning.
- **Save/Load Model**: persists policy model in localStorage.
- **Spawn Object**: spawn box/sphere/ramp near camera focus.
- **Follow Agent**: cycles follow camera target.
- **Shift + drag** an object to reposition it.
