import {
  ASSETS,
  applyTransform,
  buildGraph,
  buildPointCloud,
  buildTrajectory,
  compactNumber,
  createCameraSync,
  createViewport,
  fetchJson,
  formatTime,
  loadAlignmentConfig,
  resizeViewport,
  timestampToSeconds,
  updateGraphPlayback,
  updateTrajectory,
  upperBound,
} from "./viewer_core.js?v=tmp-main-pipeline-20s";

const dom = {
  demoShell: document.querySelector(".demo-shell"),
  leftResizer: document.querySelector("#leftResizer"),
  rightResizer: document.querySelector("#rightResizer"),
  graphCanvas: document.querySelector("#graphCanvas"),
  denseCanvas: document.querySelector("#denseCanvas"),
  loading: document.querySelector("#loadingOverlay"),
  loadingText: document.querySelector("#loadingText"),
  originalFrameVideo: document.querySelector("#originalFrameVideo"),
  originalFrameImage: document.querySelector("#originalFrameImage"),
  originalFrameStatus: document.querySelector("#originalFrameStatus"),
  trackingCanvas: document.querySelector("#trackingCanvas"),
  trackingVideo: document.querySelector("#trackingVideo"),
  depthVideo: document.querySelector("#depthVideo"),
  playToggle: document.querySelector("#playToggle"),
  timeline: document.querySelector("#timeline"),
  clock: document.querySelector("#clock"),
  speedSelect: document.querySelector("#speedSelect"),
  labelsToggle: document.querySelector("#labelsToggle"),
  pointSize: document.querySelector("#pointSize"),
  fpsBadge: document.querySelector("#fpsBadge"),
  graphPhaseLabel: document.querySelector("#graphPhaseLabel"),
  graphCameraLabel: document.querySelector("#graphCameraLabel"),
  densePhaseLabel: document.querySelector("#densePhaseLabel"),
  denseCameraLabel: document.querySelector("#denseCameraLabel"),
  nodeCount: document.querySelector("#nodeCount"),
  edgeCount: document.querySelector("#edgeCount"),
  pointCount: document.querySelector("#pointCount"),
  questionList: document.querySelector("#questionList"),
};

const state = {
  keys: new Set(),
  progress: 0,
  maxVideoDuration: 20,
  graphMaxTime: 20,
  pointMaxFrame: 1,
  dynamicPointCloud: false,
  pointWindowSeconds: 8,
  denseMapError: null,
  lastFrameTime: performance.now(),
  lastTrackingPaint: 0,
  originalFrames: [],
  depthFrames: [],
  trackingFrames: [],
  lastOriginalFrameIndex: -1,
  lastDepthFrameIndex: -1,
  lastTrackingFrameIndex: -1,
  trackingFrameRequest: 0,
  questions: [],
  qwenResults: [],
  activeResizer: null,
};

const PANEL_LAYOUT_KEY = "directme-demo-panel-layout-v1";
const MIN_LEFT_PANEL = 300;
const MIN_CENTER_PANEL = 520;
const MIN_RIGHT_PANEL = 320;

const graphViewport = createViewport(dom.graphCanvas);
const denseViewport = createViewport(dom.denseCanvas);
const cameraSync = createCameraSync([graphViewport, denseViewport]);
let graphAssets = null;
let pointCloud = null;
let denseTrajectory = null;

function loadPanelLayout() {
  try {
    const raw = window.localStorage.getItem(PANEL_LAYOUT_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (Number.isFinite(data.left)) {
      dom.demoShell.style.setProperty("--left-panel-width", `${data.left}px`);
    }
    if (Number.isFinite(data.right)) {
      dom.demoShell.style.setProperty("--right-panel-width", `${data.right}px`);
    }
  } catch (error) {
    console.warn("Failed to load saved panel layout.", error);
  }
}

function savePanelLayout(left, right) {
  try {
    window.localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify({ left, right }));
  } catch (error) {
    console.warn("Failed to save panel layout.", error);
  }
}

function bindPanelResizers() {
  const startResize = (side, event) => {
    if (window.innerWidth <= 1180) return;
    event.preventDefault();
    state.activeResizer = side;
    document.body.classList.add("is-resizing");
    window.addEventListener("pointermove", onResizePointerMove);
    window.addEventListener("pointerup", stopResize);
  };

  dom.leftResizer.addEventListener("pointerdown", (event) => startResize("left", event));
  dom.rightResizer.addEventListener("pointerdown", (event) => startResize("right", event));
}

function onResizePointerMove(event) {
  if (!state.activeResizer) return;
  const shellRect = dom.demoShell.getBoundingClientRect();
  const resizerWidth = parseFloat(getComputedStyle(dom.demoShell).getPropertyValue("--resizer-width")) || 14;
  const totalGap = resizerWidth * 2;
  const leftRect = dom.leftResizer.getBoundingClientRect();
  const rightRect = dom.rightResizer.getBoundingClientRect();
  let nextLeft = leftRect.left - shellRect.left;
  let nextRight = shellRect.right - rightRect.right;

  if (state.activeResizer === "left") {
    nextLeft = event.clientX - shellRect.left - resizerWidth / 2;
  } else {
    nextRight = shellRect.right - event.clientX - resizerWidth / 2;
  }

  const maxLeft = Math.max(MIN_LEFT_PANEL, shellRect.width - MIN_RIGHT_PANEL - MIN_CENTER_PANEL - totalGap);
  nextLeft = clamp(nextLeft, MIN_LEFT_PANEL, maxLeft);
  const maxRight = Math.max(MIN_RIGHT_PANEL, shellRect.width - nextLeft - MIN_CENTER_PANEL - totalGap);
  nextRight = clamp(nextRight, MIN_RIGHT_PANEL, maxRight);

  dom.demoShell.style.setProperty("--left-panel-width", `${nextLeft}px`);
  dom.demoShell.style.setProperty("--right-panel-width", `${nextRight}px`);
  savePanelLayout(nextLeft, nextRight);
  resize();
}

function stopResize() {
  state.activeResizer = null;
  document.body.classList.remove("is-resizing");
  window.removeEventListener("pointermove", onResizePointerMove);
  window.removeEventListener("pointerup", stopResize);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function setLoading(text) {
  dom.loadingText.textContent = text;
}

function hideLoading() {
  dom.loading.classList.add("hidden");
}

function resize() {
  resizeViewport(graphViewport);
  resizeViewport(denseViewport);
  renderTrackingFrame(true);
}

async function loadOriginalFrames() {
  const data = await fetchJson("/api/original-frames", { frames: [] });
  state.originalFrames = Array.isArray(data.frames) ? data.frames : [];

  if (state.originalFrames.length && dom.originalFrameImage) {
    dom.originalFrameImage.src = state.originalFrames[0];
    dom.originalFrameImage.style.display = "block";
    if (dom.originalFrameVideo) dom.originalFrameVideo.style.display = "none";
    dom.originalFrameStatus.classList.remove("visible");
    state.lastOriginalFrameIndex = 0;
    return;
  }

  if (dom.originalFrameVideo) {
    dom.originalFrameVideo.style.display = "block";
    if (dom.originalFrameImage) dom.originalFrameImage.style.display = "none";
    dom.originalFrameStatus.textContent = "Loading original video...";
    dom.originalFrameStatus.classList.add("visible");
    dom.originalFrameVideo.load();
    return;
  }

  dom.originalFrameStatus.textContent = "Original frames are unavailable.";
  dom.originalFrameStatus.classList.add("visible");
}

async function loadPerceptionFrames() {
  const [depthData, trackingData] = await Promise.all([
    fetchJson("/api/depth-frames", { frames: [] }),
    fetchJson("/api/tracking-frames", { frames: [] }),
  ]);
  state.depthFrames = Array.isArray(depthData.frames) ? depthData.frames : [];
  state.trackingFrames = Array.isArray(trackingData.frames) ? trackingData.frames : [];

  if (state.depthFrames.length) {
    const depthImage = ensureDepthFrameImage();
    depthImage.src = state.depthFrames[0];
    depthImage.style.display = "block";
    dom.depthVideo.style.display = "none";
    dom.depthVideo.pause();
    state.lastDepthFrameIndex = 0;
  }

  if (state.trackingFrames.length) {
    dom.trackingVideo.pause();
    state.lastTrackingFrameIndex = -1;
    renderTrackingFrame(true);
  }
}

function usingOriginalFrameImages() {
  return Boolean(dom.originalFrameImage && state.originalFrames.length);
}

function ensureDepthFrameImage() {
  let image = document.querySelector("#depthFrameImage");
  if (image) return image;
  image = document.createElement("img");
  image.id = "depthFrameImage";
  image.alt = "Depth map frame";
  image.style.width = "100%";
  image.style.height = "100%";
  image.style.objectFit = "contain";
  image.style.display = "none";
  dom.depthVideo.insertAdjacentElement("afterend", image);
  return image;
}

function frameIndexForSeconds(seconds, count) {
  if (!count) return -1;
  return Math.max(0, Math.min(count - 1, Math.floor(seconds)));
}

function frameIndexForProgress(progress, count) {
  return frameIndexForSeconds(progress * state.maxVideoDuration, count);
}

function updateOriginalMedia(progress) {
  if (usingOriginalFrameImages()) {
    const index = frameIndexForProgress(progress, state.originalFrames.length);
    if (index === state.lastOriginalFrameIndex) return;
    dom.originalFrameImage.src = state.originalFrames[index];
    state.lastOriginalFrameIndex = index;
    return;
  }

  const video = dom.originalFrameVideo;
  if (!video || !Number.isFinite(video.duration) || video.duration <= 0) return;
  const seconds = Math.max(0, Math.min(video.duration, progress * state.maxVideoDuration));
  if (Math.abs(video.currentTime - seconds) < 0.04) return;
  video.currentTime = seconds;
}

function updateDepthMedia(seconds) {
  if (!state.depthFrames.length) return;
  const index = frameIndexForSeconds(seconds, state.depthFrames.length);
  if (index === state.lastDepthFrameIndex) return;
  ensureDepthFrameImage().src = state.depthFrames[index];
  state.lastDepthFrameIndex = index;
}

function syncMediaToTime(seconds) {
  dom.trackingVideo.currentTime = seconds;
  if (!state.depthFrames.length) dom.depthVideo.currentTime = seconds;
  if (usingOriginalFrameImages()) {
    updateOriginalMedia(seconds / state.maxVideoDuration);
  } else if (dom.originalFrameVideo && Number.isFinite(dom.originalFrameVideo.duration)) {
    dom.originalFrameVideo.currentTime = Math.max(0, Math.min(dom.originalFrameVideo.duration, seconds));
  }
  updateDepthMedia(seconds);
  updateFromProgress(seconds / state.maxVideoDuration);
  renderTrackingFrame(true);
}

function syncMaxDurationFromMedia() {
  const candidates = [
    state.graphMaxTime,
    dom.trackingVideo?.duration,
    dom.depthVideo?.duration,
    dom.originalFrameVideo?.duration,
    state.originalFrames.length ? state.originalFrames.length : 0,
    state.depthFrames.length ? state.depthFrames.length : 0,
    state.trackingFrames.length ? state.trackingFrames.length : 0,
  ];
  const duration = Math.max(
    1,
    ...candidates.filter((value) => Number.isFinite(value) && value > 0),
  );
  if (Math.abs(duration - state.maxVideoDuration) > 0.01) {
    state.maxVideoDuration = duration;
    updateFromProgress(state.progress);
  }
  return state.maxVideoDuration;
}

function jumpToTime(seconds) {
  const nextTime = Math.min(seconds, state.maxVideoDuration);
  dom.trackingVideo.pause();
  if (!state.depthFrames.length) dom.depthVideo.pause();
  dom.playToggle.textContent = "Play";
  updateQwenFps(0);
  syncMediaToTime(nextTime);
}

function updateFromProgress(progress) {
  state.progress = Math.max(0, Math.min(1, progress));
  const seconds = state.progress * state.maxVideoDuration;
  const graphTimeLimit = state.progress * state.graphMaxTime;
  const pointFrameLimit = state.progress * state.pointMaxFrame;
  updateOriginalMedia(state.progress);
  updateDepthMedia(seconds);

  if (graphAssets) {
    updateGraphPlayback(graphAssets, graphTimeLimit, {
      complete: state.progress >= 0.995,
      labels: dom.labelsToggle.checked,
    });
    const sample = updateTrajectory(graphAssets.trajectory, seconds);
    if (sample) dom.graphCameraLabel.textContent = `Frame ${sample.frame_index ?? 0}`;
  }

  if (pointCloud) {
    updatePointCloudWindow(pointCloud, pointFrameLimit);
  }

  const denseSample = updateTrajectory(denseTrajectory, seconds);
  if (denseSample) dom.denseCameraLabel.textContent = `Frame ${denseSample.frame_index ?? 0}`;
  updateTimelineUi(seconds);
  updateQuestionState(seconds);
  updateQwenFps(seconds);
}

function updatePointCloudWindow(cloud, currentFrame) {
  if (!state.dynamicPointCloud) {
    let visiblePoints = upperBound(cloud.frames, currentFrame);
    if (state.progress >= 0.995) visiblePoints = cloud.count;
    cloud.geometry.setDrawRange(0, visiblePoints);
    return;
  }

  const halfWindow = state.pointWindowSeconds / 2;
  const startFrame = Math.max(0, currentFrame - halfWindow);
  const endFrame = Math.min(cloud.maxFrame, currentFrame + halfWindow);
  const start = upperBound(cloud.frames, startFrame - 0.001);
  const end = upperBound(cloud.frames, endFrame);
  cloud.geometry.setDrawRange(start, Math.max(0, end - start));
}

function updateTimelineUi(seconds) {
  dom.timeline.value = String(Math.round(state.progress * Number(dom.timeline.max)));
  dom.clock.textContent = `${formatTime(seconds)} / ${formatTime(state.maxVideoDuration)}`;
  const phase = state.progress >= 0.995 ? "Mapping Complete" : `Mapping ${Math.round(state.progress * 100)}%`;
  dom.graphPhaseLabel.textContent = phase;
  dom.densePhaseLabel.textContent = state.denseMapError
    ? "Dense map asset missing"
    : state.dynamicPointCloud
      ? `Dynamic window ${state.pointWindowSeconds}s`
      : phase;
}

function renderQuestions(items) {
  state.questions = items
    .map((item) => ({
      ...item,
      seconds: timestampToSeconds(item.question_timestamps ?? item.query_timestamp ?? 0),
    }))
    .sort((a, b) => a.seconds - b.seconds);

  dom.questionList.replaceChildren(
    ...state.questions.map((item, index) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      const answer = questionAnswer(item);
      button.className = "question-item";
      button.type = "button";
      button.dataset.index = String(index);
      button.innerHTML = `
        <span class="question-time">${formatTime(item.seconds)}</span>
        <span class="question-copy">
          <strong>${escapeHtml(item.question || item.question_chinese || "Untitled question")}</strong>
          <span class="question-meta">${escapeHtml(item.qid || item.q_id || `question_${index + 1}`)}</span>
          ${answer ? `<span class="question-answer"><em>Answer</em>${escapeHtml(answer)}</span>` : ""}
        </span>
      `;
      button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        jumpToTime(item.seconds);
      });
      button.addEventListener("click", (event) => {
        event.preventDefault();
      });
      li.append(button);
      return li;
    }),
  );
}

function questionAnswer(item) {
  for (const key of ["answer", "reference_answer", "raw_answer", "answer_chinese"]) {
    const value = String(item?.[key] ?? "").trim();
    if (value) return value;
  }

  const label = item?.answer_label ?? item?.predicted_label;
  const option = label && item?.options ? item.options[label] : "";
  return option ? `${label}. ${option}` : "";
}

function updateQuestionState(seconds) {
  const buttons = [...dom.questionList.querySelectorAll(".question-item")];
  if (!buttons.length) return;
  let activeIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  state.questions.forEach((item, index) => {
    const distance = Math.abs(item.seconds - seconds);
    if (distance < bestDistance) {
      bestDistance = distance;
      activeIndex = index;
    }
  });
  buttons.forEach((button, index) => {
    button.classList.toggle("active", index === activeIndex);
    button.classList.toggle("answer-visible", seconds >= state.questions[index].seconds);
  });
}

async function loadQwenResults() {
  const data = await fetchJson(ASSETS.qwenResults, { results: [] });
  state.qwenResults = Array.isArray(data.results) ? data.results : [];
  updateQwenFps(0);
}

function updateQwenFps(seconds) {
  if (dom.trackingVideo.paused || !state.qwenResults.length) {
    dom.fpsBadge.textContent = "FPS 0.00";
    return;
  }

  const result = nearestQwenResult(seconds);
  const fps = Number(result?.qwen_inference?.frame_fps);
  dom.fpsBadge.textContent = `FPS ${Number.isFinite(fps) ? fps.toFixed(2) : "0.00"}`;
}

function nearestQwenResult(seconds) {
  let best = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const result of state.qwenResults) {
    const timestamp = Number(result.timestamp_s);
    if (!Number.isFinite(timestamp)) continue;
    const distance = Math.abs(timestamp - seconds);
    if (distance < bestDistance) {
      best = result;
      bestDistance = distance;
    }
  }
  return best;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return entities[char];
  });
}

function renderTrackingFrame(force = false) {
  if (state.trackingFrames.length) {
    renderTrackingImageFrame(force);
    return;
  }

  const video = dom.trackingVideo;
  const canvas = dom.trackingCanvas;
  if (!video || !canvas || video.readyState < 2) return;
  const now = performance.now();
  if (!force && now - state.lastTrackingPaint < 58) return;
  state.lastTrackingPaint = now;

  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, width, height);
  thickenTrackingOverlays(ctx, width, height);
}

function renderTrackingImageFrame(force = false) {
  const canvas = dom.trackingCanvas;
  if (!canvas) return;
  const seconds = state.progress * state.maxVideoDuration;
  const index = frameIndexForSeconds(seconds, state.trackingFrames.length);
  if (!force && index === state.lastTrackingFrameIndex) return;
  state.lastTrackingFrameIndex = index;

  const requestId = state.trackingFrameRequest + 1;
  state.trackingFrameRequest = requestId;
  const image = new Image();
  image.onload = () => {
    if (requestId !== state.trackingFrameRequest) return;
    const width = image.naturalWidth || 1280;
    const height = image.naturalHeight || 720;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, width, height);
    thickenTrackingOverlays(ctx, width, height);
  };
  image.src = state.trackingFrames[index];
}

function thickenTrackingOverlays(ctx, width, height) {
  const image = ctx.getImageData(0, 0, width, height);
  const src = image.data;
  const out = new Uint8ClampedArray(src);
  const radius = 2;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const r = src[i];
      const g = src[i + 1];
      const b = src[i + 2];
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      if (max < 135 || max - min < 42) continue;

      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          if (Math.abs(dx) + Math.abs(dy) > radius) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const j = (ny * width + nx) * 4;
          out[j] = Math.max(out[j], r);
          out[j + 1] = Math.max(out[j + 1], g);
          out[j + 2] = Math.max(out[j + 2], b);
          out[j + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(new ImageData(out, width, height), 0, 0);
}

function bindUi() {
  dom.playToggle.addEventListener("click", () => {
    if (dom.trackingVideo.paused) {
      dom.trackingVideo.play();
      if (!state.depthFrames.length) dom.depthVideo.play();
    } else {
      dom.trackingVideo.pause();
      if (!state.depthFrames.length) dom.depthVideo.pause();
    }
  });

  dom.trackingVideo.addEventListener("play", () => {
    dom.playToggle.textContent = "Pause";
    if (!state.depthFrames.length && dom.depthVideo.paused) dom.depthVideo.play();
    updateQwenFps(dom.trackingVideo.currentTime || 0);
  });

  dom.trackingVideo.addEventListener("pause", () => {
    dom.playToggle.textContent = "Play";
    if (!state.depthFrames.length && !dom.depthVideo.paused) dom.depthVideo.pause();
    updateQwenFps(0);
  });

  dom.trackingVideo.addEventListener("timeupdate", () => {
    const duration = syncMaxDurationFromMedia();
    updateFromProgress(dom.trackingVideo.currentTime / duration);
    if (!state.depthFrames.length && Math.abs(dom.depthVideo.currentTime - dom.trackingVideo.currentTime) > 0.12) {
      dom.depthVideo.currentTime = dom.trackingVideo.currentTime;
    }
    renderTrackingFrame();
  });

  dom.trackingVideo.addEventListener("loadedmetadata", () => {
    syncMaxDurationFromMedia();
    updateFromProgress(0);
    renderTrackingFrame(true);
  });

  dom.depthVideo.addEventListener("loadedmetadata", () => {
    syncMaxDurationFromMedia();
  });

  dom.trackingVideo.addEventListener("seeked", () => renderTrackingFrame(true));
  dom.trackingVideo.addEventListener("loadeddata", () => renderTrackingFrame(true));

  if (dom.originalFrameVideo) {
    dom.originalFrameVideo.addEventListener("loadedmetadata", () => {
      dom.originalFrameStatus.classList.remove("visible");
      syncMaxDurationFromMedia();
      updateOriginalMedia(state.progress);
    });

    dom.originalFrameVideo.addEventListener("loadeddata", () => {
      dom.originalFrameStatus.classList.remove("visible");
    });

    dom.originalFrameVideo.addEventListener("error", () => {
      if (usingOriginalFrameImages()) return;
      dom.originalFrameStatus.textContent = "Original video failed to load.";
      dom.originalFrameStatus.classList.add("visible");
    });
  }

  dom.timeline.addEventListener("input", () => {
    const progress = Number(dom.timeline.value) / Number(dom.timeline.max);
    const nextTime = progress * state.maxVideoDuration;
    syncMediaToTime(nextTime);
  });

  dom.speedSelect.addEventListener("change", () => {
    const rate = Number(dom.speedSelect.value);
    dom.trackingVideo.playbackRate = rate;
    dom.depthVideo.playbackRate = rate;
  });

  dom.labelsToggle.addEventListener("change", () => updateFromProgress(state.progress));
  dom.pointSize.addEventListener("input", () => {
    if (pointCloud) pointCloud.material.size = Number(dom.pointSize.value);
  });

  window.addEventListener("resize", resize);
  window.addEventListener("keydown", (event) => state.keys.add(event.key.toLowerCase()));
  window.addEventListener("keyup", (event) => state.keys.delete(event.key.toLowerCase()));
}

function animate(now = performance.now()) {
  const delta = Math.min(0.05, (now - state.lastFrameTime) / 1000);
  state.lastFrameTime = now;
  cameraSync.updateMovement(state.keys, delta);
  if (!dom.trackingVideo.paused && !dom.trackingVideo.ended) {
    const duration = syncMaxDurationFromMedia();
    updateFromProgress(dom.trackingVideo.currentTime / duration);
    if (!state.depthFrames.length && Math.abs(dom.depthVideo.currentTime - dom.trackingVideo.currentTime) > 0.12) {
      dom.depthVideo.currentTime = dom.trackingVideo.currentTime;
    }
  }
  graphViewport.controls.update();
  denseViewport.controls.update();
  renderTrackingFrame();
  graphViewport.renderer.render(graphViewport.scene, graphViewport.camera);
  denseViewport.renderer.render(denseViewport.scene, denseViewport.camera);
  requestAnimationFrame(animate);
}

async function init() {
  loadPanelLayout();
  bindPanelResizers();
  bindUi();
  resize();

  setLoading("Loading alignment config...");
  const alignment = await loadAlignmentConfig();
  applyTransform(graphViewport.trajectoryRoot, alignment.graph);
  applyTransform(denseViewport.trajectoryRoot, alignment.graph);
  applyTransform(graphViewport.graphRoot, alignment.sceneGraph);
  applyTransform(graphViewport.labelsRoot, alignment.sceneGraph);
  applyTransform(denseViewport.sceneRoot, alignment.scene);

  setLoading("Loading scene graph...");
  const graph = await fetchJson(ASSETS.graph);
  graphAssets = buildGraph(graphViewport, graph, { labels: true });
  denseTrajectory = buildTrajectory(denseViewport, graph.metadata?.ego_pose_timeline || []);
  state.graphMaxTime = graphAssets.maxTime;
  syncMaxDurationFromMedia();
  dom.nodeCount.textContent = String(graphAssets.nodeRecords.length);
  dom.edgeCount.textContent = String(graphAssets.edgeRecords.length);

  setLoading("Parsing RGB semantic point cloud...");
  try {
    pointCloud = await buildPointCloud(denseViewport, { graph, pointSize: Number(dom.pointSize.value) });
    state.pointMaxFrame = Math.max(1, pointCloud.maxFrame);
    dom.pointCount.textContent = compactNumber(pointCloud.count);
  } catch (error) {
    console.warn(error);
    state.denseMapError = error.message || "Dense map asset missing";
    dom.pointCount.textContent = "missing";
    dom.densePhaseLabel.textContent = "Dense map asset missing";
    dom.denseCameraLabel.textContent = "Export dense PLY";
  }

  setLoading("Loading original video frames...");
  await loadOriginalFrames();
  await loadPerceptionFrames();
  syncMaxDurationFromMedia();

  setLoading("Loading questions...");
  await loadQwenResults();
  renderQuestions(await fetchJson(ASSETS.qa, []));
  updateFromProgress(0);
  hideLoading();
  animate();
}

init().catch((error) => {
  console.error(error);
  setLoading(error.message || "Failed to load demo assets.");
});
