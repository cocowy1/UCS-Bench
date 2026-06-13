import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export { THREE };

const DEMO_ROOT = "/tmp/directme_scal3r_full_pipeline";

export const ASSETS = {
  graph: `${DEMO_ROOT}/directme_mapping_run/scene_graph.json`,
  pointCloud: `${DEMO_ROOT}/dense_pointcloud_world.ply`,
  semanticPointCloud: `${DEMO_ROOT}/semantic_pointcloud_fused.ply`,
  pointCloudMetadata: `${DEMO_ROOT}/semantic_objects_fused.json`,
  qa: "/directme/demo/sample_qa.json",
  qwenResults: "/directme/demo/qwen_scene0804_results.json",
  alignment: "/directme/demo/web/alignment_config.json",
};

export const DEFAULT_ALIGNMENT = {
  version: 1,
  graph: { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: 1 },
  sceneGraph: { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: 1 },
  scene: { position: [0, 0, 0], rotationDeg: [0, 0, 0], scale: 1 },
};

const COLOR_MAP = {
  red: 0xec7f73,
  orange: 0xe6a45f,
  yellow: 0xe5cf64,
  green: 0x7fc8a9,
  blue: 0x86afd8,
  cyan: 0x6fd1d4,
  purple: 0xb89ce5,
  pink: 0xe493b5,
  brown: 0xb19072,
  gray: 0x9aa4a0,
  black: 0x3d4541,
  white: 0xe7ece8,
};

export async function fetchJson(url, fallback = null) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return await response.json();
  } catch (error) {
    if (fallback !== null) return fallback;
    throw error;
  }
}

export async function loadAlignmentConfig() {
  const data = await fetchJson(`${ASSETS.alignment}?t=${Date.now()}`, DEFAULT_ALIGNMENT);
  return normalizeAlignment(data);
}

export function normalizeAlignment(config) {
  const out = structuredClone(DEFAULT_ALIGNMENT);
  for (const key of ["graph", "sceneGraph", "scene"]) {
    const source = config?.[key] || {};
    out[key].position = normalizeVector(source.position, out[key].position);
    out[key].rotationDeg = normalizeVector(source.rotationDeg, out[key].rotationDeg);
    out[key].scale = Number.isFinite(Number(source.scale)) ? Number(source.scale) : out[key].scale;
  }
  return out;
}

function normalizeVector(value, fallback) {
  if (!Array.isArray(value) || value.length < 3) return [...fallback];
  return value.slice(0, 3).map((item, index) => {
    const next = Number(item);
    return Number.isFinite(next) ? next : fallback[index];
  });
}

export function applyTransform(group, transform) {
  group.position.set(...transform.position);
  group.rotation.set(
    THREE.MathUtils.degToRad(transform.rotationDeg[0]),
    THREE.MathUtils.degToRad(transform.rotationDeg[1]),
    THREE.MathUtils.degToRad(transform.rotationDeg[2]),
  );
  group.scale.setScalar(transform.scale);
}

export function createViewport(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x0e120f, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0e120f, 9, 26);

  const camera = new THREE.PerspectiveCamera(62, 1, 0.01, 120);
  camera.position.set(0, 1.45, 4.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.2, 0.15, 0.4);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.92;
  controls.minDistance = 0.4;
  controls.maxDistance = 22;

  const contentRoot = new THREE.Group();
  const graphRoot = new THREE.Group();
  const sceneRoot = new THREE.Group();
  const labelsRoot = new THREE.Group();
  const trajectoryRoot = new THREE.Group();

  scene.add(contentRoot);
  contentRoot.add(sceneRoot, graphRoot, labelsRoot, trajectoryRoot);
  addSceneHelpers(scene, contentRoot);

  return {
    canvas,
    renderer,
    scene,
    camera,
    controls,
    contentRoot,
    graphRoot,
    sceneRoot,
    labelsRoot,
    trajectoryRoot,
  };
}

function addSceneHelpers(scene, root) {
  const hemi = new THREE.HemisphereLight(0xcde7dc, 0x1f251f, 1.8);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 1.7);
  key.position.set(3.5, 5, 4);
  scene.add(key);

  const grid = new THREE.GridHelper(10, 20, 0x60736a, 0x2b3430);
  grid.position.y = -0.02;
  root.add(grid);

  const axes = new THREE.AxesHelper(0.8);
  axes.position.set(-4.4, 0.02, -4.4);
  root.add(axes);
}

export function resizeViewport(viewport) {
  const rect = viewport.canvas.parentElement.getBoundingClientRect();
  viewport.renderer.setSize(rect.width, rect.height, false);
  viewport.camera.aspect = rect.width / Math.max(1, rect.height);
  viewport.camera.updateProjectionMatrix();
}

export function createCameraSync(viewports) {
  let active = viewports[0];
  let syncing = false;

  const syncFrom = (source) => {
    if (syncing) return;
    syncing = true;
    for (const viewport of viewports) {
      if (viewport === source) continue;
      viewport.camera.position.copy(source.camera.position);
      viewport.camera.quaternion.copy(source.camera.quaternion);
      viewport.controls.target.copy(source.controls.target);
      viewport.controls.update();
    }
    syncing = false;
  };

  for (const viewport of viewports) {
    viewport.canvas.addEventListener("pointerdown", () => {
      active = viewport;
    });
    viewport.controls.addEventListener("change", () => syncFrom(viewport));
  }

  const updateMovement = (keys, delta) => {
    const speed = (keys.has("shift") ? 4.8 : 2.2) * delta;
    const forward = new THREE.Vector3();
    active.camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, active.camera.up).normalize().multiplyScalar(-1);
    const move = new THREE.Vector3();

    if (keys.has("w")) move.add(forward);
    if (keys.has("s")) move.sub(forward);
    if (keys.has("d")) move.add(right);
    if (keys.has("a")) move.sub(right);
    if (keys.has(" ")) move.y += 1;
    if (keys.has("control")) move.y -= 1;

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed);
      active.camera.position.add(move);
      active.controls.target.add(move);
      syncFrom(active);
    }
  };

  return { syncFrom, updateMovement };
}

export function toWorldVector(values) {
  // Mirror X for the observer-facing viewer so video right/left stays aligned
  // across ego heading, scene graph positions, and dense world points.
  return new THREE.Vector3(-values[0], -values[1], values[2]);
}

export function colorForNode(node) {
  const name = String(node.attributes?.color || "").toLowerCase();
  if (COLOR_MAP[name]) return COLOR_MAP[name];
  let hash = 0;
  for (const ch of node.semantic_label || node.node_id) {
    hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  }
  const hue = (hash % 360) / 360;
  return new THREE.Color().setHSL(hue, 0.48, 0.58).getHex();
}

export function observationCount(node) {
  return Number(node.spatial_absolute?.observation_count || node.observations?.length || 0);
}

export function firstSeenSeconds(node) {
  const obs = Array.isArray(node.observations) ? node.observations : [];
  if (!obs.length) return 0;
  const firstSeen = obs.reduce((minValue, item) => {
    const ts = Number(item.timestamp);
    return Number.isFinite(ts) ? Math.min(minValue, ts) : minValue;
  }, Number.POSITIVE_INFINITY);
  return Number.isFinite(firstSeen) ? firstSeen : 0;
}

export function buildGraph(viewport, graph, { labels = true } = {}) {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : Object.values(graph.nodes || {});
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const timeline = graph.metadata?.ego_pose_timeline || [];
  const nodeById = new Map();
  const nodeRecords = [];
  const edgeRecords = [];
  const materialCache = new Map();
  const sphereGeometry = new THREE.SphereGeometry(0.055, 20, 14);

  for (const node of nodes) {
    if (isHiddenGraphNode(node)) continue;
    const finalPosition = toWorldVector(node.spatial_absolute?.p_world || node.p_world || [0, 0, 0]);
    const observationTimeline = buildNodeObservationTimeline(node);
    const position = observationTimeline.positions[0]?.clone() || finalPosition.clone();
    const color = colorForNode(node);
    if (!materialCache.has(color)) {
      materialCache.set(
        color,
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.18,
          metalness: 0.08,
          roughness: 0.6,
        }),
      );
    }
    const mesh = new THREE.Mesh(sphereGeometry, materialCache.get(color));
    mesh.position.copy(position);
    mesh.scale.setScalar(1 + Math.min(observationCount(node), 60) / 80);
    viewport.graphRoot.add(mesh);

    const label = makeLabel(node.semantic_label || node.node_id, color);
    label.position.copy(position).add(new THREE.Vector3(0, 0.16, 0));
    label.visible = labels;
    viewport.labelsRoot.add(label);

    const record = {
      node,
      mesh,
      label,
      firstSeen: firstSeenSeconds(node),
      position,
      finalPosition,
      observationTimeline,
    };
    nodeRecords.push(record);
    nodeById.set(node.node_id, record);
  }

  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x8aa49a,
    transparent: true,
    opacity: 0.22,
  });

  for (const edge of edges) {
    if (edge.relation !== "near" || !nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    const geometry = new THREE.BufferGeometry().setFromPoints([source.position, target.position]);
    const line = new THREE.Line(geometry, edgeMaterial);
    viewport.graphRoot.add(line);
    edgeRecords.push({ edge, line, source, target, firstSeen: Math.max(source.firstSeen, target.firstSeen) });
  }

  return {
    nodeRecords,
    edgeRecords,
    trajectory: buildTrajectory(viewport, timeline),
    maxTime: Math.max(
      1,
      ...timeline.map((item) => Number(item.timestamp) || 0),
      ...nodes.flatMap((node) => (node.observations || []).map((obs) => Number(obs.timestamp) || 0)),
    ),
  };
}

function isHiddenGraphNode(node) {
  return String(node.semantic_label || "").trim().toLowerCase() === "picture/frame";
}

function buildNodeObservationTimeline(node) {
  const records = [];
  for (const observation of Array.isArray(node.observations) ? node.observations : []) {
    const timestamp = Number(observation.timestamp);
    const point = observation.p_world;
    if (!Number.isFinite(timestamp) || !Array.isArray(point) || point.length < 3) continue;
    records.push({ timestamp, position: toWorldVector(point) });
  }
  records.sort((a, b) => a.timestamp - b.timestamp);
  return {
    times: Float32Array.from(records.map((record) => record.timestamp)),
    positions: records.map((record) => record.position),
  };
}

export function updateGraphPlayback(graphAssets, seconds, { complete = false, labels = true } = {}) {
  if (!graphAssets) return;

  for (const record of graphAssets.nodeRecords) {
    const visible = complete || record.firstSeen <= seconds;
    record.mesh.visible = visible;
    record.label.visible = visible && labels;
    record.position.copy(nodePositionAt(record, seconds, complete));
    record.mesh.position.copy(record.position);
    record.label.position.copy(record.position);
    record.label.position.y += 0.16;
  }

  for (const record of graphAssets.edgeRecords) {
    record.line.visible = complete || record.firstSeen <= seconds;
    updateEdgeLine(record);
  }
}

function nodePositionAt(record, seconds, complete) {
  if (complete || !record.observationTimeline.times.length) return record.finalPosition;
  const index = Math.max(0, upperBound(record.observationTimeline.times, seconds) - 1);
  return record.observationTimeline.positions[Math.min(index, record.observationTimeline.positions.length - 1)];
}

function updateEdgeLine(record) {
  const positions = record.line.geometry.getAttribute("position");
  positions.setXYZ(0, record.source.position.x, record.source.position.y, record.source.position.z);
  positions.setXYZ(1, record.target.position.x, record.target.position.y, record.target.position.z);
  positions.needsUpdate = true;
}

export function buildTrajectory(viewport, timeline) {
  const points = timeline.map((item) => toWorldVector(item.translation || [0, 0, 0]));
  if (points.length < 2) return null;

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0xe5b75e });
  const line = new THREE.Line(geometry, material);
  viewport.trajectoryRoot.add(line);

  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), points[0], 0.32, 0xe5b75e, 0.13, 0.08);
  viewport.trajectoryRoot.add(arrow);

  return { timeline, points, line, arrow };
}

export function updateTrajectory(trajectory, seconds) {
  if (!trajectory) return null;
  const timestamps = trajectory.timeline.map((item) => Number(item.timestamp) || 0);
  const idx = Math.max(1, upperBound(Float32Array.from(timestamps), seconds));
  trajectory.line.geometry.setDrawRange(0, Math.min(idx, trajectory.points.length));
  const sample = trajectory.timeline[Math.min(idx - 1, trajectory.timeline.length - 1)];
  if (!sample) return null;
  const position = toWorldVector(sample.translation || [0, 0, 0]);
  trajectory.arrow.position.copy(position);
  trajectory.arrow.setDirection(toViewerHeading(sample.T_world_from_camera));
  return sample;
}

function toViewerHeading(matrix) {
  if (!Array.isArray(matrix)) return new THREE.Vector3(0, 0, 1);
  return toWorldVector([matrix[0][2], matrix[1][2], matrix[2][2]]).normalize();
}

export async function buildPointCloud(viewport, { graph = null, pointSize = 0.018, budget = 190000 } = {}) {
  const metadata = await fetchJson(ASSETS.pointCloudMetadata, []);
  const ply = await fetchPointCloudText(metadata, graph);
  const parsed = parseAsciiPly(ply.text, buildPointFrameLookup(metadata), budget);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(parsed.colors, 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    transparent: true,
    opacity: 0.78,
    sizeAttenuation: true,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  viewport.sceneRoot.add(points);
  return { points, geometry, material, frames: parsed.frames, count: parsed.count, maxFrame: parsed.maxFrame };
}

async function fetchText(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return await response.text();
}

async function fetchPointCloudText(metadata, graph) {
  try {
    return { source: ASSETS.pointCloud, text: await fetchText(ASSETS.pointCloud) };
  } catch (_error) {
    assertPointCloudTimelineMatchesGraph(ASSETS.semanticPointCloud, metadata, graph);
    return { source: ASSETS.semanticPointCloud, text: await fetchText(ASSETS.semanticPointCloud) };
  }
}

function assertPointCloudTimelineMatchesGraph(source, metadata, graph) {
  const graphFrames = graph?.metadata?.ego_pose_timeline || [];
  const graphMaxFrame = Math.max(0, ...graphFrames.map((item) => Number(item.frame_index) || 0));
  const pointFrames = (Array.isArray(metadata) ? metadata : []).flatMap((item) => item.frames || [item.frame_index]);
  const pointMaxFrame = Math.max(0, ...pointFrames.map((item) => Number(item) || 0));
  if (graphMaxFrame > pointMaxFrame + 1) {
    throw new Error(
      `Dense map asset mismatch: ${source} covers frames 0-${pointMaxFrame}, scene graph covers 0-${graphMaxFrame}. `
      + "Export dense_pointcloud_world.ply from the current SCAL3R run.",
    );
  }
}

export function buildPointFrameLookup(metadata) {
  const idToFrame = new Map();
  for (const item of Array.isArray(metadata) ? metadata : []) {
    const observationId = Number(item.observation_id);
    const frameIndex = Number(item.frame_index);
    if (Number.isFinite(observationId)) {
      idToFrame.set(observationId, Number.isFinite(frameIndex) ? frameIndex : 0);
    }

    const fusedId = Number(item.fused_id);
    if (!Number.isFinite(observationId) && Number.isFinite(fusedId)) {
      idToFrame.set(fusedId, normalizeFrames(item.frames, frameIndex));
    }
  }
  return idToFrame;
}

function normalizeFrames(frames, fallback) {
  const values = [];
  for (const value of Array.isArray(frames) ? frames : []) {
    const frame = Number(value);
    if (Number.isFinite(frame)) values.push(frame);
  }
  if (values.length) return [...new Set(values)].sort((a, b) => a - b);
  return [Number.isFinite(fallback) ? fallback : 0];
}

export function parseAsciiPly(text, idToFrame, budget) {
  const endToken = "end_header";
  const headerEnd = text.indexOf(endToken);
  if (headerEnd < 0) throw new Error("PLY header is missing end_header");

  const headerText = text.slice(0, headerEnd + endToken.length);
  const headerLines = headerText.split(/\r?\n/);
  const properties = [];
  let vertexCount = 0;
  let inVertex = false;

  for (const line of headerLines) {
    if (line.startsWith("element vertex")) {
      vertexCount = Number(line.split(/\s+/)[2]);
      inVertex = true;
      continue;
    }
    if (line.startsWith("element ") && !line.startsWith("element vertex")) inVertex = false;
    if (inVertex && line.startsWith("property ")) properties.push(line.trim().split(/\s+/)[2]);
  }

  const ix = properties.indexOf("x");
  const iy = properties.indexOf("y");
  const iz = properties.indexOf("z");
  const ir = properties.indexOf("red");
  const ig = properties.indexOf("green");
  const ib = properties.indexOf("blue");
  const iframe = properties.indexOf("frame_index");
  const pointIdIndex = properties.findIndex((name) => name === "observation_id" || name === "fused_id");
  const dataStart = text.indexOf("\n", headerEnd) + 1;
  const lines = text.slice(dataStart).trim().split(/\r?\n/);
  const stride = Math.max(1, Math.ceil(vertexCount / budget));
  const rows = [];

  for (let i = 0; i < lines.length; i += stride) {
    const values = lines[i].trim().split(/\s+/);
    if (values.length < properties.length) continue;
    const pointId = pointIdIndex >= 0 ? Number(values[pointIdIndex]) : 0;
    const frame = iframe >= 0 ? Number(values[iframe]) || 0 : 0;
    rows.push({
      x: -Number(values[ix]),
      y: -Number(values[iy]),
      z: Number(values[iz]),
      r: Number(values[ir]) / 255,
      g: Number(values[ig]) / 255,
      b: Number(values[ib]) / 255,
      pointId,
      frame,
    });
  }

  if (iframe < 0) assignPointFrames(rows, idToFrame);
  rows.sort((a, b) => a.frame - b.frame);
  const positions = new Float32Array(rows.length * 3);
  const colors = new Float32Array(rows.length * 3);
  const frames = new Float32Array(rows.length);
  let maxFrame = 0;

  rows.forEach((row, index) => {
    const offset = index * 3;
    positions[offset] = row.x;
    positions[offset + 1] = row.y;
    positions[offset + 2] = row.z;
    colors[offset] = row.r;
    colors[offset + 1] = row.g;
    colors[offset + 2] = row.b;
    frames[index] = row.frame;
    maxFrame = Math.max(maxFrame, row.frame);
  });

  return { positions, colors, frames, count: rows.length, maxFrame };
}

function assignPointFrames(rows, idToFrame) {
  const pointCounts = new Map();
  for (const row of rows) {
    pointCounts.set(row.pointId, (pointCounts.get(row.pointId) || 0) + 1);
  }

  const pointIndices = new Map();
  for (const row of rows) {
    const frameSpec = idToFrame.get(row.pointId) ?? 0;
    if (!Array.isArray(frameSpec)) {
      row.frame = frameSpec;
      continue;
    }

    const index = pointIndices.get(row.pointId) || 0;
    const count = pointCounts.get(row.pointId) || 1;
    // Fused PLY rows keep object ids but not per-observation ids. Split each
    // fused point block over that object's observed frames for playback.
    const frameIndex = Math.min(frameSpec.length - 1, Math.floor((index * frameSpec.length) / count));
    row.frame = frameSpec[frameIndex];
    pointIndices.set(row.pointId, index + 1);
  }
}

export function upperBound(values, target) {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function makeLabel(text, color) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const maxWidth = 280;
  canvas.width = 512;
  canvas.height = 128;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "700 34px Aptos, Segoe UI, sans-serif";
  ctx.textBaseline = "middle";
  const label = text.length > 24 ? `${text.slice(0, 22)}...` : text;
  const measured = Math.min(maxWidth, ctx.measureText(label).width + 34);
  ctx.fillStyle = "rgba(12, 17, 14, 0.72)";
  roundRect(ctx, 18, 32, measured, 58, 14);
  ctx.fill();
  ctx.strokeStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = "#eef4ef";
  ctx.fillText(label, 36, 62, maxWidth - 22);
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(0.85, 0.22, 1);
  return sprite;
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

export function compactNumber(value) {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${Math.round(value / 1000)}k`;
  return String(value);
}

export function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const mm = String(Math.floor(safe / 60)).padStart(2, "0");
  const ss = String(Math.floor(safe % 60)).padStart(2, "0");
  return `${mm}:${ss}`;
}

export function timestampToSeconds(value) {
  if (typeof value === "number") return value;
  const parts = String(value || "0").split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return Number(parts[0]) || 0;
}
