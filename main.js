import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import * as CANNON from 'https://cdn.jsdelivr.net/npm/cannon-es@0.20.0/dist/cannon-es.js';

import { createWorld } from './world.js';
import { HumanoidAgent } from './humanoid.js';
import { MultiAgentTrainer } from './training.js';
import { createUI } from './ui.js';

const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

const { scene, world, waterMesh, isInWater, spawnObject, syncDynamicMeshes, dynamicObjects, terrainHeight } = createWorld();

const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 700);
camera.position.set(16, 13, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 2, 0);
controls.enableDamping = true;

const agentColors = [0xf97316, 0x22c55e, 0x3b82f6, 0xa855f7, 0xe11d48];
const agents = [];
for (let i = 0; i < 5; i += 1) {
  const x = -8 + i * 3.5;
  const z = -2 + (i % 2) * 4;
  const y = terrainHeight(x, z) + 0.5;
  agents.push(new HumanoidAgent(i, world, scene, agentColors[i], new CANNON.Vec3(x, y, z), terrainHeight));
}

const trainer = new MultiAgentTrainer(agents);

let followIndex = -1;
const ui = createUI({
  trainer,
  agents,
  onResetAll: () => {
    agents.forEach((a, idx) => {
      const x = -8 + idx * 3.5;
      const z = -2 + (idx % 2) * 4;
      const y = terrainHeight(x, z) + 0.5;
      a.reset(new CANNON.Vec3(x, y, z));
    });
  },
  onSpawnObject: (kind) => {
    const p = new CANNON.Vec3(controls.target.x, controls.target.y + 8, controls.target.z);
    spawnObject(kind, p);
  },
  onToggleFollow: () => {
    followIndex = (followIndex + 1) % (agents.length + 1) - 1;
  },
});

// Drag support in editor mode (hold Shift).
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let dragObj = null;
let dragOffset = new THREE.Vector3();

function pointerToNdc(e) {
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!e.shiftKey) return;
  pointerToNdc(e);
  raycaster.setFromCamera(pointer, camera);
  const meshes = dynamicObjects.map((o) => o.mesh);
  const hit = raycaster.intersectObjects(meshes)[0];
  if (!hit) return;

  dragObj = dynamicObjects.find((o) => o.mesh === hit.object);
  const p = new THREE.Vector3();
  raycaster.ray.intersectPlane(dragPlane, p);
  dragOffset.copy(p).sub(dragObj.mesh.position);
  controls.enabled = false;
});

window.addEventListener('pointermove', (e) => {
  if (!dragObj) return;
  pointerToNdc(e);
  raycaster.setFromCamera(pointer, camera);
  const p = new THREE.Vector3();
  raycaster.ray.intersectPlane(dragPlane, p);
  p.sub(dragOffset);
  dragObj.body.position.set(p.x, Math.max(terrainHeight(p.x, p.z) + 1, p.y), p.z);
  dragObj.body.velocity.set(0, 0, 0);
  dragObj.body.angularVelocity.set(0, 0, 0);
});

window.addEventListener('pointerup', () => {
  dragObj = null;
  controls.enabled = true;
});

const fixedTimeStep = 1 / 60;
const maxSubSteps = 4;
let last = performance.now() / 1000;

function loop() {
  requestAnimationFrame(loop);

  const now = performance.now() / 1000;
  const dt = Math.min(0.06, now - last);
  last = now;

  trainer.step();
  world.step(fixedTimeStep, dt, maxSubSteps);

  for (const agent of agents) {
    // Water physics effect: reduced gravity and additional drag.
    if (isInWater(agent.torsoBody)) {
      for (const body of Object.values(agent.bodies)) {
        body.applyForce(new CANNON.Vec3(0, body.mass * 5.3, 0));
        body.velocity.scale(0.96, body.velocity);
        body.angularVelocity.scale(0.92, body.angularVelocity);
      }
    }

    agent.updateMetrics(dt);
    agent.syncMeshes();
  }

  waterMesh.material.uniforms.time.value += dt;
  syncDynamicMeshes();

  if (followIndex >= 0) {
    const t = agents[followIndex].torsoBody.position;
    controls.target.lerp(new THREE.Vector3(t.x, t.y + 0.4, t.z), 0.08);
  }

  controls.update();
  ui.renderStats();
  renderer.render(scene, camera);
}

loop();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
