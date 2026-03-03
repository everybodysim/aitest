import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import * as CANNON from 'https://cdn.jsdelivr.net/npm/cannon-es@0.20.0/dist/cannon-es.js';

const PART_DEFS = [
  { key: 'pelvis', size: [0.45, 0.2, 0.25], offset: [0, 1.15, 0], mass: 3.8 },
  { key: 'torso', size: [0.45, 0.4, 0.25], offset: [0, 1.62, 0], mass: 5.6 },
  { key: 'chest', size: [0.42, 0.3, 0.24], offset: [0, 2.03, 0], mass: 4.6 },
  { key: 'neck', size: [0.12, 0.12, 0.12], offset: [0, 2.32, 0], mass: 0.8 },
  { key: 'head', size: [0.2, 0.25, 0.2], offset: [0, 2.62, 0], mass: 2.2 },

  { key: 'lClav', size: [0.18, 0.08, 0.08], offset: [-0.34, 2.12, 0], mass: 0.6 },
  { key: 'lUpperArm', size: [0.13, 0.28, 0.12], offset: [-0.58, 1.88, 0], mass: 1.5 },
  { key: 'lForeArm', size: [0.11, 0.26, 0.11], offset: [-0.58, 1.5, 0], mass: 1.1 },
  { key: 'lHand', size: [0.1, 0.07, 0.13], offset: [-0.58, 1.24, 0], mass: 0.6 },

  { key: 'rClav', size: [0.18, 0.08, 0.08], offset: [0.34, 2.12, 0], mass: 0.6 },
  { key: 'rUpperArm', size: [0.13, 0.28, 0.12], offset: [0.58, 1.88, 0], mass: 1.5 },
  { key: 'rForeArm', size: [0.11, 0.26, 0.11], offset: [0.58, 1.5, 0], mass: 1.1 },
  { key: 'rHand', size: [0.1, 0.07, 0.13], offset: [0.58, 1.24, 0], mass: 0.6 },

  { key: 'lThigh', size: [0.15, 0.33, 0.15], offset: [-0.2, 0.74, 0], mass: 2.4 },
  { key: 'lShin', size: [0.13, 0.31, 0.13], offset: [-0.2, 0.28, 0], mass: 1.9 },
  { key: 'lFoot', size: [0.11, 0.06, 0.24], offset: [-0.2, 0.05, 0.1], mass: 0.95 },
  { key: 'lToe', size: [0.09, 0.04, 0.13], offset: [-0.2, 0.03, 0.34], mass: 0.35 },

  { key: 'rThigh', size: [0.15, 0.33, 0.15], offset: [0.2, 0.74, 0], mass: 2.4 },
  { key: 'rShin', size: [0.13, 0.31, 0.13], offset: [0.2, 0.28, 0], mass: 1.9 },
  { key: 'rFoot', size: [0.11, 0.06, 0.24], offset: [0.2, 0.05, 0.1], mass: 0.95 },
  { key: 'rToe', size: [0.09, 0.04, 0.13], offset: [0.2, 0.03, 0.34], mass: 0.35 },
];

const JOINTS = [
  ['spine0', 'pelvis', 'torso', [0, 1.36, 0], [1, 0, 0], 55],
  ['spine1', 'torso', 'chest', [0, 1.85, 0], [1, 0, 0], 45],
  ['neck', 'chest', 'neck', [0, 2.22, 0], [1, 0, 0], 30],
  ['head', 'neck', 'head', [0, 2.44, 0], [1, 0, 0], 25],

  ['lShoulder0', 'chest', 'lClav', [-0.2, 2.12, 0], [0, 0, 1], 35],
  ['lShoulder1', 'lClav', 'lUpperArm', [-0.46, 2.0, 0], [1, 0, 0], 45],
  ['lElbow', 'lUpperArm', 'lForeArm', [-0.58, 1.68, 0], [1, 0, 0], 35],
  ['lWrist', 'lForeArm', 'lHand', [-0.58, 1.34, 0], [1, 0, 0], 20],

  ['rShoulder0', 'chest', 'rClav', [0.2, 2.12, 0], [0, 0, -1], 35],
  ['rShoulder1', 'rClav', 'rUpperArm', [0.46, 2.0, 0], [1, 0, 0], 45],
  ['rElbow', 'rUpperArm', 'rForeArm', [0.58, 1.68, 0], [1, 0, 0], 35],
  ['rWrist', 'rForeArm', 'rHand', [0.58, 1.34, 0], [1, 0, 0], 20],

  ['lHip', 'pelvis', 'lThigh', [-0.2, 0.96, 0], [1, 0, 0], 60],
  ['lKnee', 'lThigh', 'lShin', [-0.2, 0.48, 0], [1, 0, 0], 55],
  ['lAnkle', 'lShin', 'lFoot', [-0.2, 0.11, 0.06], [1, 0, 0], 30],
  ['lToe', 'lFoot', 'lToe', [-0.2, 0.03, 0.23], [1, 0, 0], 15],

  ['rHip', 'pelvis', 'rThigh', [0.2, 0.96, 0], [1, 0, 0], 60],
  ['rKnee', 'rThigh', 'rShin', [0.2, 0.48, 0], [1, 0, 0], 55],
  ['rAnkle', 'rShin', 'rFoot', [0.2, 0.11, 0.06], [1, 0, 0], 30],
  ['rToe', 'rFoot', 'rToe', [0.2, 0.03, 0.23], [1, 0, 0], 15],
];

export class HumanoidAgent {
  constructor(id, world, scene, color, spawn, terrainHeightFn) {
    this.id = id;
    this.world = world;
    this.scene = scene;
    this.bodies = {};
    this.meshes = {};
    this.joints = [];
    this.start = spawn.clone();
    this.terrainHeight = terrainHeightFn;

    this.material = new THREE.MeshStandardMaterial({ color, roughness: 0.74, metalness: 0.06 });
    this.resetStats();

    this.#buildBodies();
    this.#buildJoints();
    this.reset(spawn);
  }

  #buildBodies() {
    for (const part of PART_DEFS) {
      const half = new CANNON.Vec3(part.size[0], part.size[1], part.size[2]);
      const body = new CANNON.Body({ mass: part.mass, shape: new CANNON.Box(half), angularDamping: 0.2, linearDamping: 0.08 });
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(part.size[0] * 2, part.size[1] * 2, part.size[2] * 2),
        this.material,
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.world.addBody(body);
      this.scene.add(mesh);
      this.bodies[part.key] = body;
      this.meshes[part.key] = mesh;
    }
  }

  #buildJoints() {
    for (const [name, a, b, worldPivot, axis, maxTorque] of JOINTS) {
      const bodyA = this.bodies[a];
      const bodyB = this.bodies[b];
      const pivot = new CANNON.Vec3(...worldPivot);
      const pivotA = pivot.vsub(bodyA.position);
      const pivotB = pivot.vsub(bodyB.position);
      const c = new CANNON.HingeConstraint(bodyA, bodyB, {
        pivotA,
        pivotB,
        axisA: new CANNON.Vec3(...axis),
        axisB: new CANNON.Vec3(...axis),
        collideConnected: false,
      });
      c.enableMotor();
      c.setMotorSpeed(0);
      c.motorEquation.maxForce = maxTorque;
      c.motorEquation.minForce = -maxTorque;
      this.world.addConstraint(c);
      this.joints.push({ name, constraint: c, maxTorque });
    }
  }

  get observationSize() {
    return this.joints.length * 2 + 10;
  }

  get actionSize() {
    return this.joints.length;
  }

  get torsoBody() {
    return this.bodies.chest;
  }

  resetStats() {
    this.stats = {
      reward: 0,
      timeAlive: 0,
      distance: 0,
      lastX: 0,
    };
  }

  reset(at = this.start) {
    this.resetStats();
    for (const part of PART_DEFS) {
      const body = this.bodies[part.key];
      body.velocity.set(0, 0, 0);
      body.angularVelocity.set(0, 0, 0);
      body.quaternion.set(0, 0, 0, 1);
      body.position.set(at.x + part.offset[0], at.y + part.offset[1], at.z + part.offset[2]);
      body.wakeUp();
    }
    this.stats.lastX = this.torsoBody.position.x;
  }

  applyActions(actions) {
    for (let i = 0; i < this.joints.length; i += 1) {
      const j = this.joints[i];
      const torque = THREE.MathUtils.clamp(actions[i] || 0, -1, 1) * j.maxTorque;
      j.constraint.setMotorSpeed(torque * 0.02);
      j.constraint.motorEquation.maxForce = Math.abs(torque);
      j.constraint.motorEquation.minForce = -Math.abs(torque);
    }
  }

  getObservations() {
    const obs = [];
    for (const j of this.joints) {
      const axis = j.constraint.axisA;
      const relV = j.constraint.bodyB.angularVelocity.vsub(j.constraint.bodyA.angularVelocity);
      const v = relV.dot(axis) * 0.1;
      obs.push(Math.tanh(v));
      const qA = j.constraint.bodyA.quaternion;
      const qB = j.constraint.bodyB.quaternion;
      const dot = qA.x * qB.x + qA.y * qB.y + qA.z * qB.z + qA.w * qB.w;
      obs.push(dot);
    }

    const torso = this.torsoBody;
    obs.push(torso.position.y * 0.2);
    obs.push(torso.velocity.x * 0.2, torso.velocity.y * 0.2, torso.velocity.z * 0.2);

    const up = new CANNON.Vec3(0, 1, 0);
    const localUp = torso.quaternion.vmult(up);
    obs.push(localUp.x, localUp.y, localUp.z);
    obs.push((torso.position.x - this.start.x) * 0.05, (torso.position.z - this.start.z) * 0.05, this.stats.timeAlive * 0.01);

    return obs;
  }

  updateMetrics(dt) {
    const torso = this.torsoBody;
    this.stats.timeAlive += dt;
    this.stats.distance += Math.max(0, torso.position.x - this.stats.lastX);
    this.stats.lastX = torso.position.x;

    const floorY = this.terrainHeight(torso.position.x, torso.position.z);
    if (torso.position.y < floorY - 1.6) {
      this.reset();
    }
  }

  syncMeshes() {
    for (const key of Object.keys(this.meshes)) {
      this.meshes[key].position.copy(this.bodies[key].position);
      this.meshes[key].quaternion.copy(this.bodies[key].quaternion);
    }
  }
}
