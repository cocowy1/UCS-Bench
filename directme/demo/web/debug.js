import {
  ASSETS,
  DEFAULT_ALIGNMENT,
  applyTransform,
  buildGraph,
  buildPointCloud,
  buildTrajectory,
  createCameraSync,
  createViewport,
  fetchJson,
  formatTime,
  loadAlignmentConfig,
  normalizeAlignment,
  resizeViewport,
  updateGraphPlayback,
  updateTrajectory,
  upperBound,
} from "./viewer_core.js";

const dom = {
  graphCanvas: document.querySelector("#graphCanvas"),
  denseCanvas: document.querySelector("#denseCanvas"),
  loading: document.querySelector("#loadingOverlay"),
  loadingText: document.querySelector("#loadingText"),
  labelsToggle: document.querySelector("#labelsToggle"),
  pointSize: document.querySelector("#pointSize"),
  debugTimeline: document.querySelector("#debugTimeline"),
  debugClock: document.querySelector("#debugClock"),
  graphControls: document.querySelector("#graphControls"),
  sceneGraphControls: document.querySelector("#sceneGraphControls"),
  sceneControls: document.querySelector("#sceneControls"),
  saveConfig: document.querySelector("#saveConfig"),
  resetConfig: document.querySelector("#resetConfig"),
  flipSceneY: document.querySelector("#flipSceneY"),
  statusLine: document.querySelector("#statusLine"),
  graphCameraLabel: document.querySelector("#graphCameraLabel"),
  denseCameraLabel: document.querySelector("#denseCameraLabel"),
};

const state = {
  keys: new Set(),
  config: structuredClone(DEFAULT_ALIGNMENT),
  progress: 1,
  maxTime: 20,
  graphMaxTime: 20,
  pointMaxFrame: 1,
  lastFrameTime: performance.now(),
};

const graphViewport = createViewport(dom.graphCanvas);
const denseViewport = createViewport(dom.denseCanvas);
const cameraSync = createCameraSync([graphViewport, denseViewport]);
let graphAssets = null;
let denseTrajectory = null;
let pointCloud = null;

const SLIDERS = [
  { label: "X", group: "position", index: 0, min: -6, max: 6, step: 0.01 },
  { label: "Y", group: "position", index: 1, min: -6, max: 6, step: 0.01 },
  { label: "Z", group: "position", index: 2, min: -6, max: 6, step: 0.01 },
  { label: "Rot X", group: "rotationDeg", index: 0, min: -180, max: 180, step: 1 },
  { label: "Rot Y", group: "rotationDeg", index: 1, min: -180, max: 180, step: 1 },
  { label: "Rot Z", group: "rotationDeg", index: 2, min: -180, max: 180, step: 1 },
  { label: "Scale", group: "scale", index: null, min: 0.05, max: 5, step: 0.01 },
];

function setLoading(text) {
  dom.loadingText.textContent = text;
}

function hideLoading() {
  dom.loading.classList.add("hidden");
}

function setStatus(text) {
  dom.statusLine.textContent = text;
}

function resize() {
  resizeViewport(graphViewport);
  resizeViewport(denseViewport);
}

function renderControls() {
  dom.graphControls.replaceChildren(...SLIDERS.map((slider) => createSlider("graph", slider)));
  dom.sceneGraphControls.replaceChildren(...SLIDERS.map((slider) => createSlider("sceneGraph", slider)));
  dom.sceneControls.replaceChildren(...SLIDERS.map((slider) => createSlider("scene", slider)));
}

function createSlider(target, slider) {
  const label = document.createElement("label");
  label.className = "slider-row";
  const input = document.createElement("input");
  const output = document.createElement("output");
  input.type = "range";
  input.min = String(slider.min);
  input.max = String(slider.max);
  input.step = String(slider.step);
  input.dataset.target = target;
  input.dataset.group = slider.group;
  input.dataset.index = slider.index === null ? "" : String(slider.index);
  input.value = String(getConfigValue(target, slider));
  output.value = input.value;
  output.textContent = formatSliderValue(Number(input.value), slider);
  input.addEventListener("input", () => {
    setConfigValue(target, slider, Number(input.value));
    output.textContent = formatSliderValue(Number(input.value), slider);
    applyCurrentTransforms();
  });
  label.append(document.createElement("span"), input, output);
  label.firstElementChild.textContent = slider.label;
  return label;
}

function getConfigValue(target, slider) {
  if (slider.group === "scale") return state.config[target].scale;
  return state.config[target][slider.group][slider.index];
}

function setConfigValue(target, slider, value) {
  if (slider.group === "scale") state.config[target].scale = value;
  else state.config[target][slider.group][slider.index] = value;
}

function formatSliderValue(value, slider) {
  if (slider.group === "rotationDeg") return `${Math.round(value)}deg`;
  if (slider.group === "scale") return value.toFixed(2);
  return value.toFixed(2);
}

function applyCurrentTransforms() {
  applyTransform(graphViewport.trajectoryRoot, state.config.graph);
  applyTransform(denseViewport.trajectoryRoot, state.config.graph);
  applyTransform(graphViewport.graphRoot, state.config.sceneGraph);
  applyTransform(graphViewport.labelsRoot, state.config.sceneGraph);
  applyTransform(denseViewport.sceneRoot, state.config.scene);
}

function updateFromProgress(progress) {
  state.progress = Math.max(0, Math.min(1, progress));
  const seconds = state.progress * state.maxTime;
  const graphTimeLimit = state.progress * state.graphMaxTime;
  const pointFrameLimit = state.progress * state.pointMaxFrame;

  if (graphAssets) {
    updateGraphPlayback(graphAssets, graphTimeLimit, {
      complete: state.progress >= 0.995,
      labels: dom.labelsToggle.checked,
    });
    const sample = updateTrajectory(graphAssets.trajectory, seconds);
    if (sample) dom.graphCameraLabel.textContent = `Frame ${sample.frame_index ?? 0}`;
  }

  if (pointCloud) {
    let visiblePoints = upperBound(pointCloud.frames, state.progress * state.pointMaxFrame);
    if (state.progress >= 0.995) visiblePoints = pointCloud.count;
    pointCloud.geometry.setDrawRange(0, visiblePoints);
  }

  const denseSample = updateTrajectory(denseTrajectory, seconds);
  if (denseSample) dom.denseCameraLabel.textContent = `Frame ${denseSample.frame_index ?? 0}`;
  dom.debugTimeline.value = String(Math.round(state.progress * Number(dom.debugTimeline.max)));
  dom.debugClock.textContent = formatTime(seconds);
}

function bindUi() {
  dom.debugTimeline.addEventListener("input", () => {
    updateFromProgress(Number(dom.debugTimeline.value) / Number(dom.debugTimeline.max));
  });
  dom.labelsToggle.addEventListener("change", () => updateFromProgress(state.progress));
  dom.pointSize.addEventListener("input", () => {
    if (pointCloud) pointCloud.material.size = Number(dom.pointSize.value);
  });
  dom.resetConfig.addEventListener("click", () => {
    state.config = structuredClone(DEFAULT_ALIGNMENT);
    renderControls();
    applyCurrentTransforms();
    setStatus("Reset to defaults. Click Save Config to write the file.");
  });
  dom.flipSceneY.addEventListener("click", () => {
    state.config.scene.rotationDeg[1] = wrapDegrees(state.config.scene.rotationDeg[1] + 180);
    renderControls();
    applyCurrentTransforms();
    setStatus("Applied a 180deg Y-axis flip to Dense Mapping.");
  });
  dom.saveConfig.addEventListener("click", saveConfig);
  window.addEventListener("resize", resize);
  window.addEventListener("keydown", (event) => state.keys.add(event.key.toLowerCase()));
  window.addEventListener("keyup", (event) => state.keys.delete(event.key.toLowerCase()));
}

function wrapDegrees(value) {
  let next = value;
  while (next > 180) next -= 360;
  while (next < -180) next += 360;
  return next;
}

async function saveConfig() {
  setStatus("Saving alignment_config.json...");
  try {
    const response = await fetch("/api/alignment-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(normalizeAlignment(state.config), null, 2),
    });
    if (!response.ok) throw new Error(`save failed: ${response.status}`);
    setStatus("Saved. Refresh the main demo page to load this config.");
  } catch (error) {
    console.error(error);
    setStatus("Save failed. Start the server with python3 -m directme.demo.web_server.");
  }
}

function animate(now = performance.now()) {
  const delta = Math.min(0.05, (now - state.lastFrameTime) / 1000);
  state.lastFrameTime = now;
  cameraSync.updateMovement(state.keys, delta);
  graphViewport.controls.update();
  denseViewport.controls.update();
  graphViewport.renderer.render(graphViewport.scene, graphViewport.camera);
  denseViewport.renderer.render(denseViewport.scene, denseViewport.camera);
  requestAnimationFrame(animate);
}

async function init() {
  bindUi();
  resize();

  setLoading("Loading alignment config...");
  state.config = await loadAlignmentConfig();
  renderControls();
  applyCurrentTransforms();

  setLoading("Loading scene graph...");
  const graph = await fetchJson(ASSETS.graph);
  graphAssets = buildGraph(graphViewport, graph, { labels: true });
  denseTrajectory = buildTrajectory(denseViewport, graph.metadata?.ego_pose_timeline || []);
  state.graphMaxTime = graphAssets.maxTime;
  state.maxTime = Math.max(1, graphAssets.maxTime);

  setLoading("Parsing RGB semantic point cloud...");
  try {
    pointCloud = await buildPointCloud(denseViewport, { graph, pointSize: Number(dom.pointSize.value) });
    state.pointMaxFrame = Math.max(1, pointCloud.maxFrame);
  } catch (error) {
    console.warn(error);
    setStatus(error.message || "Dense map asset is missing.");
  }

  updateFromProgress(1);
  hideLoading();
  animate();
}

init().catch((error) => {
  console.error(error);
  setLoading(error.message || "Failed to load debug assets.");
});
