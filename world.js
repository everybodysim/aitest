import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import * as CANNON from 'https://cdn.jsdelivr.net/npm/cannon-es@0.20.0/dist/cannon-es.js';

const TERRAIN_SIZE = 180;
const TERRAIN_RES = 96;
const WATER_ZONE = { x: -18, z: 22, radius: 16, surfaceY: 2.2 };

function hashNoise(x, z) {
  return Math.sin(x * 0.17) * 0.7 + Math.cos(z * 0.11) * 0.5 + Math.sin((x + z) * 0.08) * 0.3;
}

export function terrainHeight(x, z) {
  let h = hashNoise(x, z) * 2.4;

  // Flatten training plateaus.
  const plateaus = [
    { x: 12, z: 10, w: 20, d: 18, y: 3 },
    { x: -30, z: -20, w: 16, d: 16, y: 5.5 },
    { x: 36, z: -24, w: 18, d: 14, y: 7 },
  ];

  for (const p of plateaus) {
    if (Math.abs(x - p.x) < p.w * 0.5 && Math.abs(z - p.z) < p.d * 0.5) {
      h = THREE.MathUtils.lerp(h, p.y, 0.92);
    }
  }
  return h;
}

function buildHeightfield() {
  const data = [];
  for (let i = 0; i < TERRAIN_RES; i += 1) {
    const row = [];
    for (let j = 0; j < TERRAIN_RES; j += 1) {
      const x = (i / (TERRAIN_RES - 1) - 0.5) * TERRAIN_SIZE;
      const z = (j / (TERRAIN_RES - 1) - 0.5) * TERRAIN_SIZE;
      row.push(terrainHeight(x, z));
    }
    data.push(row);
  }
  return data;
}

export function createWorld() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9dd7ff);
  scene.fog = new THREE.Fog(0x9dd7ff, 80, 260);

  const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -9.82, 0) });
  world.broadphase = new CANNON.SAPBroadphase(world);
  world.allowSleep = true;

  const hemi = new THREE.HemisphereLight(0xb0e3ff, 0x3f7040, 0.8);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(40, 70, 25);
  dir.castShadow = true;
  dir.shadow.mapSize.set(2048, 2048);
  dir.shadow.camera.left = -90;
  dir.shadow.camera.right = 90;
  dir.shadow.camera.top = 90;
  dir.shadow.camera.bottom = -90;
  scene.add(dir);

  const skyGeo = new THREE.SphereGeometry(450, 24, 16);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: { top: { value: new THREE.Color(0x63b8ff) }, bottom: { value: new THREE.Color(0xe8f7ff) } },
    vertexShader: `varying vec3 vPos; void main(){ vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `uniform vec3 top; uniform vec3 bottom; varying vec3 vPos;
      void main(){ float h = normalize(vPos).y * 0.5 + 0.5; gl_FragColor = vec4(mix(bottom, top, pow(h, 1.5)), 1.0); }`,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  const heights = buildHeightfield();
  const elementSize = TERRAIN_SIZE / (TERRAIN_RES - 1);
  const hShape = new CANNON.Heightfield(heights, { elementSize });
  const groundBody = new CANNON.Body({ mass: 0, material: new CANNON.Material('ground') });
  groundBody.addShape(hShape);
  groundBody.position.set(-(TERRAIN_RES - 1) * elementSize * 0.5, 0, (TERRAIN_RES - 1) * elementSize * 0.5);
  groundBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0, 'XYZ');
  world.addBody(groundBody);

  const terrainGeo = new THREE.PlaneGeometry(TERRAIN_SIZE, TERRAIN_SIZE, TERRAIN_RES - 1, TERRAIN_RES - 1);
  terrainGeo.rotateX(-Math.PI / 2);
  const pos = terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    pos.setY(i, terrainHeight(x, z));
  }
  pos.needsUpdate = true;
  terrainGeo.computeVertexNormals();

  const terrainMat = new THREE.MeshStandardMaterial({ color: 0x5fa35f, roughness: 0.95, metalness: 0.02 });
  const terrainMesh = new THREE.Mesh(terrainGeo, terrainMat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

  const waterGeo = new THREE.CircleGeometry(WATER_ZONE.radius, 64);
  waterGeo.rotateX(-Math.PI / 2);
  const waterMat = new THREE.ShaderMaterial({
    transparent: true,
    uniforms: { time: { value: 0 } },
    vertexShader: `uniform float time; varying vec2 vUv;
      void main(){ vUv = uv; vec3 p = position; p.y += sin((p.x + time*2.0)*0.24)*0.2 + cos((p.z+time*1.7)*0.18)*0.16;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p,1.0);} `,
    fragmentShader: `varying vec2 vUv;
      void main(){ float a = 0.35 + 0.25*sin(vUv.x*24.0)*cos(vUv.y*16.0); gl_FragColor = vec4(0.12,0.45,0.72,a);} `,
  });
  const waterMesh = new THREE.Mesh(waterGeo, waterMat);
  waterMesh.position.set(WATER_ZONE.x, WATER_ZONE.surfaceY, WATER_ZONE.z);
  waterMesh.receiveShadow = true;
  scene.add(waterMesh);

  const dynamicObjects = [];

  function spawnObject(kind, position) {
    const materials = {
      box: new THREE.MeshStandardMaterial({ color: 0xc58f5c }),
      sphere: new THREE.MeshStandardMaterial({ color: 0x7da7d9 }),
      ramp: new THREE.MeshStandardMaterial({ color: 0x9f8f77 }),
    };

    let body;
    let mesh;
    if (kind === 'sphere') {
      const r = 1.1;
      body = new CANNON.Body({ mass: 5, shape: new CANNON.Sphere(r) });
      mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 14), materials.sphere);
    } else if (kind === 'ramp') {
      const half = new CANNON.Vec3(2, 0.7, 3);
      body = new CANNON.Body({ mass: 8, shape: new CANNON.Box(half) });
      mesh = new THREE.Mesh(new THREE.BoxGeometry(half.x * 2, half.y * 2, half.z * 2), materials.ramp);
      body.quaternion.setFromEuler(-0.28, 0, 0.2, 'XYZ');
      mesh.rotation.copy(new THREE.Euler(-0.28, 0, 0.2));
    } else {
      const half = new CANNON.Vec3(1, 1, 1);
      body = new CANNON.Body({ mass: 6, shape: new CANNON.Box(half) });
      mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), materials.box);
    }

    body.position.copy(position);
    body.angularDamping = 0.35;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    world.addBody(body);
    scene.add(mesh);
    dynamicObjects.push({ body, mesh });
    return { body, mesh };
  }

  function syncDynamicMeshes() {
    for (const obj of dynamicObjects) {
      obj.mesh.position.copy(obj.body.position);
      obj.mesh.quaternion.copy(obj.body.quaternion);
    }
  }

  function isInWater(body) {
    const dx = body.position.x - WATER_ZONE.x;
    const dz = body.position.z - WATER_ZONE.z;
    return dx * dx + dz * dz < WATER_ZONE.radius * WATER_ZONE.radius && body.position.y < WATER_ZONE.surfaceY + 2.5;
  }

  return {
    scene,
    world,
    waterMesh,
    isInWater,
    spawnObject,
    syncDynamicMeshes,
    dynamicObjects,
    terrainHeight,
  };
}
