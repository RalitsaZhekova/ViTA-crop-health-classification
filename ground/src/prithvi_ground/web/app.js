"use strict";

const API = "/api/v1";
const SVG_NS = "http://www.w3.org/2000/svg";

const elementIds = [
  "service-indicator", "service-label", "region-select", "theme-toggle", "refresh-scenes", "new-analysis",
  "empty-new-analysis", "empty-state", "dashboard", "region-name", "scene-title",
  "scene-subtitle", "observation-month", "observation-day", "observation-year",
  "summary-score-ring", "summary-score", "summary-condition", "summary-headline",
  "summary-message", "scene-viewer", "viewer-transform", "scene-image", "condition-image",
  "vector-heatmap", "selection-box", "viewer-hint", "cursor-coordinate", "draw-area",
  "zoom-in", "zoom-out", "reset-view", "image-resolution", "area-title", "clear-area",
  "area-score-ring", "area-score", "area-condition", "area-guidance", "area-size",
  "area-coverage", "area-alert", "area-callout", "area-callout-copy", "area-ndvi",
  "area-gndvi", "area-evi", "area-savi", "area-method-note", "history-scope",
  "history-count", "history-latest", "history-latest-label", "history-change",
  "history-change-label", "baseline-card", "baseline-status", "baseline-message", "history-chart",
  "history-strip", "evidence-score", "evidence-progress", "analysis-coverage",
  "analysis-progress", "usable-coverage", "usable-progress", "quality-list", "metric-grid",
  "scientific-claim", "manifest-link", "analysis-dialog", "analysis-form", "close-analysis",
  "analysis-fields", "analysis-sensor", "analysis-region", "analysis-input", "image-field",
  "analysis-image", "analysis-capability", "start-analysis", "analysis-progress-panel",
  "run-visual", "run-status-kicker", "run-status-title", "run-status-message", "run-elapsed",
  "run-payload-time", "view-result", "retry-analysis", "theme-color", "toast",
];

const elements = Object.fromEntries(
  elementIds.map((id) => [id, document.getElementById(id)])
);

const state = {
  scenes: [],
  history: [],
  manifests: new Map(),
  activeRegionId: null,
  activeSceneId: null,
  manifest: null,
  links: null,
  imageWidth: 1,
  imageHeight: 1,
  baseScale: 1,
  zoom: 1,
  panX: 0,
  panY: 0,
  pointerId: null,
  interaction: null,
  interactionStart: null,
  panStart: null,
  drawMode: false,
  draftPixels: null,
  selection: null,
  historyRenderToken: 0,
  pipelineCapability: null,
  pipelineRun: null,
  runPollTimer: null,
  runClockTimer: null,
  resultLoadedForRun: null,
};

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isBaselineVigorScore(value) {
  return isNumber(value) && value > 0;
}

function seasonForDate(date) {
  if (!(date instanceof Date) || Number.isNaN(date.valueOf())) return null;
  const northernSeason = Math.floor(((date.getUTCMonth() + 1) % 12) / 3);
  const bounds = state.manifest?.geospatial?.bounds_wgs84;
  const latitude = validBounds(bounds) ? (bounds[1] + bounds[3]) / 2 : 0;
  const season = latitude < 0 ? (northernSeason + 2) % 4 : northernSeason;
  return ["winter", "spring", "summer", "fall"][season];
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatNumber(value, digits = 1) {
  return isNumber(value) ? value.toFixed(digits) : "—";
}

function formatPercent(value, digits = 1) {
  return isNumber(value) ? `${value.toFixed(digits)}%` : "—";
}

function formatInteger(value) {
  return isNumber(value) ? Math.round(value).toLocaleString() : "—";
}

function humanizeIdentifier(value) {
  const words = String(value || "Unknown region").replace(/[_-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Unknown region";
}

function slug(value) {
  return String(value || "no-score")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function dateLabel(value, options = {}) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.valueOf())) return "Unknown date";
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: options.short ? "short" : "long",
    day: "numeric",
    ...(options.time ? { hour: "2-digit", minute: "2-digit", timeZoneName: "short" } : {}),
  }).format(date);
}

function setText(id, value) {
  elements[id].textContent = value;
}

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function storedTheme() {
  try {
    const value = window.localStorage.getItem("vita-theme");
    return ["light", "dark"].includes(value) ? value : null;
  } catch (_error) {
    return null;
  }
}

function applyTheme(theme, { persist = false } = {}) {
  const resolved = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
  if (persist) {
    try {
      window.localStorage.setItem("vita-theme", resolved);
    } catch (_error) {
      // Theme switching still works when storage is unavailable.
    }
  }
  const next = resolved === "dark" ? "light" : "dark";
  elements["theme-toggle"].setAttribute("aria-label", `Switch to ${next} theme`);
  elements["theme-toggle"].setAttribute("title", `Switch to ${next} theme`);
  elements["theme-toggle"].setAttribute("aria-pressed", resolved === "dark" ? "true" : "false");
  elements["theme-color"].setAttribute("content", resolved === "dark" ? "#08120e" : "#f4f7f2");
}

function toggleTheme() {
  applyTheme(currentTheme() === "dark" ? "light" : "dark", { persist: true });
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 4200);
}

async function apiFetch(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_error) {
      // The HTTP status remains useful when the response has no JSON body.
    }
    throw new Error(detail);
  }
  return response.json();
}

function conditionColor(score) {
  if (!isNumber(score)) return "#94a099";
  return interpolateStops(score, [
    [0, "#d73027"],
    [35, "#fc8d59"],
    [55, "#fee08b"],
    [75, "#91cf60"],
    [100, "#1a9850"],
  ]);
}

function interpolateStops(value, stops) {
  const score = clamp(value, stops[0][0], stops[stops.length - 1][0]);
  let left = stops[0];
  let right = stops[stops.length - 1];
  for (let index = 1; index < stops.length; index += 1) {
    if (score <= stops[index][0]) {
      left = stops[index - 1];
      right = stops[index];
      break;
    }
  }
  const ratio = right[0] === left[0] ? 0 : (score - left[0]) / (right[0] - left[0]);
  const leftRgb = hexToRgb(left[1]);
  const rightRgb = hexToRgb(right[1]);
  const channels = leftRgb.map((channel, index) =>
    Math.round(channel + (rightRgb[index] - channel) * ratio)
  );
  return `rgb(${channels.join(",")})`;
}

function hexToRgb(value) {
  const normalized = value.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(normalized.slice(offset, offset + 2), 16));
}

function labelForScore(score, configuration = {}) {
  if (!isNumber(score)) return "Insufficient data";
  if (score >= (configuration.nominal_minimum ?? 75)) return "Nominal";
  if (score >= (configuration.watch_minimum ?? 55)) return "Watch";
  if (score >= (configuration.moderate_minimum ?? 35)) return "Moderate anomaly";
  return "High anomaly";
}

function conditionCopy(score) {
  if (!isNumber(score)) {
    return {
      headline: "Not enough clear crop evidence",
      guidance: "Cloud or limited crop coverage prevents a responsible local score.",
      tone: "neutral",
    };
  }
  if (score >= 75) {
    return {
      headline: "Crop condition looks strong",
      guidance: "Satellite indicators are consistent with stronger crop vigor in this area.",
      tone: "good",
    };
  }
  if (score >= 55) {
    return {
      headline: "Keep this area under watch",
      guidance: "Condition is mixed. Compare the trend and continue monitoring before intervening.",
      tone: "neutral",
    };
  }
  if (score >= 35) {
    return {
      headline: "A closer look is recommended",
      guidance: "Satellite indicators are below the stronger range in part of this area.",
      tone: "attention",
    };
  }
  return {
    headline: "Prioritize this area for review",
    guidance: "The crop-condition signal is substantially below the stronger reference range.",
    tone: "attention",
  };
}

function setConditionPill(element, label) {
  element.textContent = label || "Insufficient data";
  element.className = `condition-pill ${slug(label)}`;
}

function setScoreRing(element, score) {
  element.style.setProperty("--score", isNumber(score) ? clamp(score, 0, 100) : 0);
  element.style.setProperty("--score-color", conditionColor(score));
}

async function checkService() {
  try {
    await apiFetch(`${API}/health`);
    elements["service-indicator"].className = "status-dot online";
    setText("service-label", "Operational");
  } catch (_error) {
    elements["service-indicator"].className = "status-dot offline";
    setText("service-label", "Unavailable");
  }
}

async function loadScenes({ targetSceneId = null, preserveSelection = true } = {}) {
  try {
    const response = await apiFetch(`${API}/scenes?limit=500`);
    state.scenes = response.items;
    if (!state.scenes.length) {
      showEmptyState();
      populateRegionSelect([]);
      return;
    }

    const regions = [...new Set(state.scenes.map((scene) => scene.region_id))].sort();
    populateRegionSelect(regions);
    const target = targetSceneId
      ? state.scenes.find((scene) => scene.scene_id === targetSceneId)
      : null;
    const regionId = target?.region_id
      || (regions.includes(state.activeRegionId) ? state.activeRegionId : state.scenes[0].region_id);
    if (!preserveSelection || (state.activeRegionId && state.activeRegionId !== regionId)) {
      state.selection = null;
    }
    await selectRegion(regionId, { targetSceneId });
  } catch (error) {
    showEmptyState();
    showToast(error.message, true);
  }
}

function populateRegionSelect(regions) {
  elements["region-select"].replaceChildren();
  if (!regions.length) {
    const option = document.createElement("option");
    option.textContent = "No regions yet";
    elements["region-select"].append(option);
    elements["region-select"].disabled = true;
    return;
  }
  elements["region-select"].disabled = false;
  regions.forEach((regionId) => {
    const option = document.createElement("option");
    option.value = regionId;
    option.textContent = humanizeIdentifier(regionId);
    elements["region-select"].append(option);
  });
}

function showEmptyState() {
  elements.dashboard.classList.add("hidden");
  elements["empty-state"].classList.remove("hidden");
}

async function selectRegion(regionId, { targetSceneId = null } = {}) {
  if (!regionId) return;
  const regionChanged = state.activeRegionId !== regionId;
  state.activeRegionId = regionId;
  elements["region-select"].value = regionId;
  if (regionChanged) state.selection = null;
  try {
    const response = await apiFetch(`${API}/regions/${encodeURIComponent(regionId)}/history`);
    if (state.activeRegionId !== regionId) return;
    state.history = response.items;
    const requested = targetSceneId
      && state.history.some((item) => item.scene_id === targetSceneId)
      ? targetSceneId
      : null;
    const nextSceneId = requested || state.history[state.history.length - 1]?.scene_id;
    await selectScene(nextSceneId, { updateHistory: false });
    await renderHistory();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function manifestForScene(scene) {
  if (state.manifests.has(scene.scene_id)) return state.manifests.get(scene.scene_id);
  const response = await apiFetch(scene.links?.manifest || `${API}/scenes/${encodeURIComponent(scene.scene_id)}/manifest`);
  const record = { manifest: response.scene, links: response.links };
  state.manifests.set(scene.scene_id, record);
  return record;
}

async function selectScene(sceneId, { updateHistory = true } = {}) {
  if (!sceneId) return;
  const scene = state.scenes.find((item) => item.scene_id === sceneId)
    || state.history.find((item) => item.scene_id === sceneId);
  if (!scene) return;
  state.activeSceneId = sceneId;
  try {
    const record = await manifestForScene(scene);
    if (state.activeSceneId !== sceneId) return;
    state.manifest = record.manifest;
    state.links = record.links;
    renderManifest(record.manifest, record.links);
    if (updateHistory) await renderHistory();
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderManifest(manifest, links) {
  elements["empty-state"].classList.add("hidden");
  elements.dashboard.classList.remove("hidden");
  setText("region-name", humanizeIdentifier(manifest.region_id));
  setText("scene-title", humanizeIdentifier(manifest.region_id));
  setText(
    "scene-subtitle",
    `${sensorLabel(manifest.sensor)} · Captured ${dateLabel(manifest.acquired_at, { time: true })}`
  );

  const acquired = new Date(manifest.acquired_at);
  if (!Number.isNaN(acquired.valueOf())) {
    setText("observation-month", new Intl.DateTimeFormat(undefined, { month: "short" }).format(acquired));
    setText("observation-day", new Intl.DateTimeFormat(undefined, { day: "2-digit" }).format(acquired));
    setText("observation-year", new Intl.DateTimeFormat(undefined, { year: "numeric" }).format(acquired));
  }

  renderAreaResult(manifest);
  renderViewer(manifest, links);
  renderObservationDetails(manifest);
}

function sensorLabel(sensor) {
  if (sensor === "sentinel-2") return "Sentinel-2";
  if (sensor === "balkan-1") return "Balkan-1";
  return humanizeIdentifier(sensor);
}

function scoreFromManifest(manifest) {
  return manifest?.condition?.condition_score;
}

function renderSummaryResult(result, selected) {
  const copy = conditionCopy(result.score);
  const evidenceMessage = selected
    ? `Selected area · based on ${formatInteger(result.analysisPixels)} clear crop pixels in this observation.`
    : `Based on ${formatInteger(result.analysisPixels)} clear crop pixels in this observation.`;
  setText("summary-score", isNumber(result.score) ? Math.round(result.score) : "—");
  setScoreRing(elements["summary-score-ring"], result.score);
  setConditionPill(elements["summary-condition"], result.label);
  setText("summary-headline", copy.headline);
  setText(
    "summary-message",
    isNumber(result.score)
      ? evidenceMessage
      : copy.guidance
  );
}

function renderViewer(manifest, links) {
  const preview = manifest.assets?.rgb_preview || {};
  state.imageWidth = preview.width || manifest.geospatial?.source_width || 1;
  state.imageHeight = preview.height || manifest.geospatial?.source_height || 1;
  elements["viewer-transform"].style.width = `${state.imageWidth}px`;
  elements["viewer-transform"].style.height = `${state.imageHeight}px`;
  elements["scene-image"].src = links.preview;
  elements["condition-image"].src = links.condition_overlay;
  renderVectorHeatmap(manifest);
  renderSelectionBox();

  const resolution = manifest.geospatial?.resolution;
  const sourceResolution = Array.isArray(resolution) && isNumber(resolution[0])
    ? ` · ${formatNumber(Math.abs(resolution[0]), 0)} m source pixels`
    : "";
  setText(
    "image-resolution",
    `${state.imageWidth.toLocaleString()} × ${state.imageHeight.toLocaleString()} display${sourceResolution}`
  );
  elements["manifest-link"].href = links.manifest;
  elements["manifest-link"].classList.remove("disabled");
  window.requestAnimationFrame(resetViewer);
}

function renderVectorHeatmap(manifest) {
  const svg = elements["vector-heatmap"];
  const grid = manifest.interaction_grid;
  svg.replaceChildren();
  if (!grid || !Number.isInteger(grid.rows) || !Number.isInteger(grid.columns)) return;
  svg.setAttribute("viewBox", `0 0 ${grid.columns} ${grid.rows}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const gradient = (manifest.legend?.condition_gradient || [])
    .filter((stop) => isNumber(stop.score) && Array.isArray(stop.rgb))
    .map((stop) => [stop.score, `#${stop.rgb.map((value) => value.toString(16).padStart(2, "0")).join("")}`]);
  grid.cells.forEach((cell) => {
    if (!isBaselineVigorScore(cell.condition_score)) return;
    const rectangle = document.createElementNS(SVG_NS, "rect");
    rectangle.setAttribute("x", cell.column);
    rectangle.setAttribute("y", cell.row);
    rectangle.setAttribute("width", "1.002");
    rectangle.setAttribute("height", "1.002");
    rectangle.setAttribute(
      "fill",
      gradient.length >= 2 ? interpolateStops(cell.condition_score, gradient) : conditionColor(cell.condition_score)
    );
    const evidence = cell.quality?.analysis_percentage;
    rectangle.setAttribute("fill-opacity", isNumber(evidence) ? clamp(0.45 + evidence / 180, 0.45, 1) : 0.55);
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `Condition ${cell.condition_score.toFixed(1)} out of 100`;
    rectangle.append(title);
    svg.append(rectangle);
  });
}

function wholeObservationResult(manifest) {
  const condition = manifest.condition || {};
  const quality = manifest.quality || {};
  const metrics = manifest.metrics || {};
  return {
    score: condition.condition_score,
    label: condition.label || labelForScore(condition.condition_score, condition.configuration),
    coverage: condition.analysis_percentage,
    alert: condition.low_vigor_percentage,
    unusable: quality.unusable_percentage,
    analysisPixels: condition.analysis_pixels,
    metrics: {
      ndvi: metrics.ndvi?.median,
      gndvi: metrics.gndvi?.median,
      evi: metrics.evi?.median,
      savi: metrics.savi?.median,
    },
    cells: manifest.interaction_grid?.cells?.length || 0,
  };
}

function renderAreaResult(manifest) {
  const selected = Boolean(state.selection);
  const result = selected
    ? aggregateSelectedArea(manifest, state.selection.boundsWgs84)
    : wholeObservationResult(manifest);
  renderSummaryResult(result, selected);
  setText("area-title", selected ? "Selected area" : "Whole observation");
  elements["clear-area"].classList.toggle("hidden", !selected);
  setText("area-score", isNumber(result.score) ? Math.round(result.score) : "—");
  setScoreRing(elements["area-score-ring"], result.score);
  setConditionPill(elements["area-condition"], result.label);

  const copy = conditionCopy(result.score);
  setText("area-guidance", copy.headline);
  elements["area-callout"].className = `area-callout ${copy.tone}`;
  setText("area-callout-copy", copy.guidance);
  setText(
    "area-size",
    selected ? formatAreaSize(geographicAreaHectares(state.selection.boundsWgs84)) : formatAreaSize(geographicAreaHectares(manifest.geospatial?.bounds_wgs84))
  );
  setText("area-coverage", formatPercent(result.coverage));
  setText("area-alert", formatPercent(result.alert));
  setText("area-ndvi", formatNumber(result.metrics?.ndvi, 3));
  setText("area-gndvi", formatNumber(result.metrics?.gndvi, 3));
  setText("area-evi", formatNumber(result.metrics?.evi, 3));
  setText("area-savi", formatNumber(result.metrics?.savi, 3));
  setText(
    "area-method-note",
    selected
      ? `Overlap-weighted from ${result.cells} payload-measured grid cell${result.cells === 1 ? "" : "s"}. Precision follows the ${manifest.interaction_grid?.rows || "—"} × ${manifest.interaction_grid?.columns || "—"} analysis grid; display colors are never sampled.`
      : "Whole-observation values come directly from the verified payload result. Display colors are never sampled."
  );
}

function aggregateSelectedArea(manifest, selectedBounds) {
  const grid = manifest.interaction_grid;
  const configuration = manifest.condition?.configuration || {};
  const empty = {
    score: null,
    label: "Insufficient data",
    coverage: 0,
    alert: null,
    unusable: null,
    analysisPixels: 0,
    metrics: {},
    cells: 0,
  };
  if (!grid?.cells?.length || !validBounds(selectedBounds)) return empty;

  const cellPixelCount = (manifest.geospatial.source_width * manifest.geospatial.source_height)
    / (grid.rows * grid.columns);
  let totalPixels = 0;
  let analysisPixels = 0;
  let scoreSum = 0;
  let scoreWeight = 0;
  let alertSum = 0;
  let alertWeight = 0;
  let unusableSum = 0;
  let totalWeight = 0;
  let cells = 0;
  const metricSums = { ndvi: 0, gndvi: 0, evi: 0, savi: 0 };
  const metricWeights = { ndvi: 0, gndvi: 0, evi: 0, savi: 0 };

  grid.cells.forEach((cell) => {
    const overlap = boundsOverlapFraction(cell.bounds_wgs84, selectedBounds);
    if (overlap <= 0) return;
    cells += 1;
    const pixelWeight = cellPixelCount * overlap;
    const measuredPixels = (cell.quality?.analysis_pixels || 0) * overlap;
    totalPixels += pixelWeight;
    totalWeight += pixelWeight;
    if (isNumber(cell.quality?.unusable_percentage)) {
      unusableSum += cell.quality.unusable_percentage * pixelWeight;
    }
    if (isBaselineVigorScore(cell.condition_score) && measuredPixels > 0) {
      analysisPixels += measuredPixels;
      scoreSum += cell.condition_score * measuredPixels;
      scoreWeight += measuredPixels;
      if (isNumber(cell.alert_percentage)) {
        alertSum += cell.alert_percentage * measuredPixels;
        alertWeight += measuredPixels;
      }
      Object.keys(metricSums).forEach((metric) => {
        if (isNumber(cell.metrics?.[metric])) {
          metricSums[metric] += cell.metrics[metric] * measuredPixels;
          metricWeights[metric] += measuredPixels;
        }
      });
    }
  });

  const score = scoreWeight > 0 ? scoreSum / scoreWeight : null;
  return {
    score,
    label: labelForScore(score, configuration),
    coverage: totalPixels > 0 ? clamp((analysisPixels / totalPixels) * 100, 0, 100) : 0,
    alert: alertWeight > 0 ? alertSum / alertWeight : null,
    unusable: totalWeight > 0 ? unusableSum / totalWeight : null,
    analysisPixels,
    metrics: Object.fromEntries(
      Object.keys(metricSums).map((metric) => [
        metric,
        metricWeights[metric] > 0 ? metricSums[metric] / metricWeights[metric] : null,
      ])
    ),
    cells,
  };
}

function validBounds(bounds) {
  return Array.isArray(bounds)
    && bounds.length === 4
    && bounds.every(isNumber)
    && bounds[0] < bounds[2]
    && bounds[1] < bounds[3];
}

function boundsOverlapFraction(cellBounds, selectedBounds) {
  if (!validBounds(cellBounds) || !validBounds(selectedBounds)) return 0;
  const width = Math.max(0, Math.min(cellBounds[2], selectedBounds[2]) - Math.max(cellBounds[0], selectedBounds[0]));
  const height = Math.max(0, Math.min(cellBounds[3], selectedBounds[3]) - Math.max(cellBounds[1], selectedBounds[1]));
  const cellArea = (cellBounds[2] - cellBounds[0]) * (cellBounds[3] - cellBounds[1]);
  return cellArea > 0 ? clamp((width * height) / cellArea, 0, 1) : 0;
}

function geographicAreaHectares(bounds) {
  if (!validBounds(bounds)) return null;
  const latitude = (bounds[1] + bounds[3]) / 2;
  const widthMeters = (bounds[2] - bounds[0]) * 111320 * Math.cos(latitude * Math.PI / 180);
  const heightMeters = (bounds[3] - bounds[1]) * 110574;
  return Math.abs(widthMeters * heightMeters) / 10000;
}

function formatAreaSize(hectares) {
  if (!isNumber(hectares)) return "—";
  if (hectares >= 1000) return `${(hectares / 100).toFixed(1)} km²`;
  if (hectares >= 10) return `${hectares.toFixed(0)} ha`;
  return `${hectares.toFixed(1)} ha`;
}

function renderObservationDetails(manifest) {
  const condition = manifest.condition || {};
  const quality = manifest.quality || {};
  setText("evidence-score", isNumber(condition.evidence_quality_score) ? `${Math.round(condition.evidence_quality_score)}/100` : "—");
  setProgress("evidence-progress", condition.evidence_quality_score);
  setText("analysis-coverage", formatPercent(condition.analysis_percentage));
  setProgress("analysis-progress", condition.analysis_percentage);
  setText("usable-coverage", formatPercent(quality.usable_percentage));
  setProgress("usable-progress", quality.usable_percentage);
  renderQuality(quality);
  renderMetrics(manifest.metrics || {});
  setText("scientific-claim", manifest.claim || "Relative crop-condition screening; not an agronomic diagnosis.");
}

function setProgress(id, value) {
  elements[id].style.width = `${clamp(isNumber(value) ? value : 0, 0, 100)}%`;
}

function renderQuality(quality) {
  const definitions = [
    ["clear_percentage", "Clear", "#3aa976"],
    ["thick_cloud_percentage", "Thick cloud", "#aeb9c1"],
    ["thin_cloud_percentage", "Thin cloud", "#52aeca"],
    ["cloud_shadow_percentage", "Cloud shadow", "#71669a"],
    ["invalid_percentage", "Unavailable", "#d5a33c"],
  ];
  elements["quality-list"].replaceChildren();
  definitions.forEach(([key, label, color]) => {
    const row = document.createElement("div");
    row.className = "quality-row";
    const name = document.createElement("span");
    const dot = document.createElement("i");
    dot.className = "quality-dot";
    dot.style.background = color;
    name.append(dot, document.createTextNode(label));
    const value = document.createElement("strong");
    value.textContent = formatPercent(quality[key]);
    row.append(name, value);
    elements["quality-list"].append(row);
  });
}

function renderMetrics(metrics) {
  const definitions = [
    ["ndvi", "Canopy vigor", "NDVI"],
    ["gndvi", "Chlorophyll", "GNDVI"],
    ["evi", "Dense canopy", "EVI"],
    ["savi", "Soil adjusted", "SAVI"],
    ["cvi", "Chlorophyll ratio", "CVI"],
    ["vari", "Visible greenness", "VARI"],
    ["excess_green", "Excess green", "EXG"],
    ["rgb_brightness", "Brightness", "RGB"],
  ];
  elements["metric-grid"].replaceChildren();
  definitions.forEach(([key, label, abbreviation]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = formatNumber(metrics[key]?.median, 3);
    const detail = document.createElement("small");
    detail.textContent = abbreviation;
    card.append(name, value, detail);
    elements["metric-grid"].append(card);
  });
}

function boundsToPixels(bounds, manifest = state.manifest) {
  const sceneBounds = manifest?.geospatial?.bounds_wgs84;
  const width = manifest?.assets?.rgb_preview?.width || state.imageWidth;
  const height = manifest?.assets?.rgb_preview?.height || state.imageHeight;
  if (!validBounds(bounds) || !validBounds(sceneBounds)) return null;
  const x1 = ((bounds[0] - sceneBounds[0]) / (sceneBounds[2] - sceneBounds[0])) * width;
  const x2 = ((bounds[2] - sceneBounds[0]) / (sceneBounds[2] - sceneBounds[0])) * width;
  const y1 = ((sceneBounds[3] - bounds[3]) / (sceneBounds[3] - sceneBounds[1])) * height;
  const y2 = ((sceneBounds[3] - bounds[1]) / (sceneBounds[3] - sceneBounds[1])) * height;
  const pixels = {
    x1: clamp(Math.min(x1, x2), 0, width),
    y1: clamp(Math.min(y1, y2), 0, height),
    x2: clamp(Math.max(x1, x2), 0, width),
    y2: clamp(Math.max(y1, y2), 0, height),
  };
  return pixels.x2 > pixels.x1 && pixels.y2 > pixels.y1 ? pixels : null;
}

function pixelsToBounds(pixels) {
  const bounds = state.manifest?.geospatial?.bounds_wgs84;
  if (!pixels || !validBounds(bounds)) return null;
  const west = bounds[0] + (pixels.x1 / state.imageWidth) * (bounds[2] - bounds[0]);
  const east = bounds[0] + (pixels.x2 / state.imageWidth) * (bounds[2] - bounds[0]);
  const north = bounds[3] - (pixels.y1 / state.imageHeight) * (bounds[3] - bounds[1]);
  const south = bounds[3] - (pixels.y2 / state.imageHeight) * (bounds[3] - bounds[1]);
  return [west, south, east, north];
}

function renderSelectionBox(pixels = null) {
  const selectionPixels = pixels || (state.selection ? boundsToPixels(state.selection.boundsWgs84) : null);
  if (!selectionPixels) {
    elements["selection-box"].classList.add("hidden");
    return;
  }
  const box = elements["selection-box"];
  box.style.left = `${selectionPixels.x1}px`;
  box.style.top = `${selectionPixels.y1}px`;
  box.style.width = `${selectionPixels.x2 - selectionPixels.x1}px`;
  box.style.height = `${selectionPixels.y2 - selectionPixels.y1}px`;
  box.classList.remove("hidden");
  updateSelectionLabelScale();
}

function updateSelectionLabelScale() {
  const label = elements["selection-box"].querySelector("span");
  if (!label) return;
  const scale = Math.max(0.01, state.baseScale * state.zoom);
  label.style.transform = `scale(${1 / scale})`;
}

function resetViewer() {
  const rect = elements["scene-viewer"].getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  state.baseScale = Math.min(
    Math.max(0.01, (rect.width - 30) / state.imageWidth),
    Math.max(0.01, (rect.height - 30) / state.imageHeight)
  );
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  applyViewerTransform();
}

function applyViewerTransform() {
  const rect = elements["scene-viewer"].getBoundingClientRect();
  const scale = state.baseScale * state.zoom;
  const x = rect.width / 2 + state.panX - (state.imageWidth * scale) / 2;
  const y = rect.height / 2 + state.panY - (state.imageHeight * scale) / 2;
  elements["viewer-transform"].style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  updateSelectionLabelScale();
}

function zoomViewer(factor, clientX = null, clientY = null) {
  const previous = state.zoom;
  const next = clamp(previous * factor, 1, 18);
  if (next === previous) return;
  const rect = elements["scene-viewer"].getBoundingClientRect();
  const pointerX = clientX === null ? rect.width / 2 : clientX - rect.left;
  const pointerY = clientY === null ? rect.height / 2 : clientY - rect.top;
  const offsetX = pointerX - rect.width / 2 - state.panX;
  const offsetY = pointerY - rect.height / 2 - state.panY;
  const ratio = next / previous;
  state.panX += offsetX * (1 - ratio);
  state.panY += offsetY * (1 - ratio);
  state.zoom = next;
  applyViewerTransform();
}

function imagePointFromEvent(event) {
  const rect = elements["scene-viewer"].getBoundingClientRect();
  const scale = state.baseScale * state.zoom;
  const x = (event.clientX - rect.left - rect.width / 2 - state.panX) / scale + state.imageWidth / 2;
  const y = (event.clientY - rect.top - rect.height / 2 - state.panY) / scale + state.imageHeight / 2;
  if (x < 0 || y < 0 || x > state.imageWidth || y > state.imageHeight) return null;
  return { x, y };
}

function updateCursorCoordinate(event) {
  const point = imagePointFromEvent(event);
  const bounds = state.manifest?.geospatial?.bounds_wgs84;
  if (!point || !validBounds(bounds)) {
    setText("cursor-coordinate", "Outside image");
    return;
  }
  const longitude = bounds[0] + (point.x / state.imageWidth) * (bounds[2] - bounds[0]);
  const latitude = bounds[3] - (point.y / state.imageHeight) * (bounds[3] - bounds[1]);
  setText("cursor-coordinate", `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`);
}

function setDrawMode(enabled) {
  state.drawMode = enabled;
  elements["draw-area"].classList.toggle("active", enabled);
  elements["draw-area"].setAttribute("aria-pressed", enabled ? "true" : "false");
  elements["scene-viewer"].classList.toggle("draw-mode", enabled);
  setText(
    "viewer-hint",
    enabled ? "Drag a box around the area you want to measure" : "Choose “Select area”, then drag a box for local results"
  );
}

function finishAreaSelection(pixels) {
  if (!pixels || pixels.x2 - pixels.x1 < 5 || pixels.y2 - pixels.y1 < 5) {
    renderSelectionBox();
    setDrawMode(false);
    return;
  }
  const bounds = pixelsToBounds(pixels);
  if (!bounds) return;
  state.selection = { boundsWgs84: bounds };
  state.draftPixels = null;
  setDrawMode(false);
  renderSelectionBox();
  renderAreaResult(state.manifest);
  renderHistory();
}

function clearAreaSelection() {
  state.selection = null;
  state.draftPixels = null;
  setDrawMode(false);
  renderSelectionBox();
  if (state.manifest) renderAreaResult(state.manifest);
  renderHistory();
}

async function renderHistory() {
  const token = ++state.historyRenderToken;
  const history = state.history.slice(-60);
  const isAreaHistory = Boolean(state.selection);
  setText(
    "history-scope",
    isAreaHistory
      ? "The same selected geographic area is compared across every available observation."
      : "Whole-region scores. Select an area on the map to follow that zone through time."
  );
  setText("history-count", `${history.length} observation${history.length === 1 ? "" : "s"}`);
  if (!history.length) {
    drawHistory([]);
    renderHistoryStrip([], new Map());
    return;
  }

  let points;
  if (isAreaHistory) {
    elements["history-chart"].replaceChildren(historyMessage("Updating selected-area history…"));
    const records = await Promise.all(
      history.map(async (scene) => {
        try {
          const record = await manifestForScene(scene);
          const result = aggregateSelectedArea(record.manifest, state.selection.boundsWgs84);
          return {
            scene,
            date: new Date(scene.acquired_at),
            score: result.score,
            label: result.label,
          };
        } catch (_error) {
          return { scene, date: new Date(scene.acquired_at), score: null, label: "Unavailable" };
        }
      })
    );
    if (token !== state.historyRenderToken) return;
    points = records;
  } else {
    points = history.map((scene) => ({
      scene,
      date: new Date(scene.acquired_at),
      score: scene.condition?.score,
      label: scene.condition?.label,
    }));
  }

  renderHistoryInsights(points);
  drawHistory(points);
  renderHistoryStrip(history, new Map(points.map((point) => [point.scene.scene_id, point])));
}

function historyMessage(message) {
  const element = document.createElement("p");
  element.className = "history-empty";
  element.textContent = message;
  return element;
}

function renderHistoryInsights(points) {
  const scored = points.filter(
    (point) => isBaselineVigorScore(point.score) && !Number.isNaN(point.date.valueOf())
  );
  const activePoint = points.find((point) => point.scene.scene_id === state.activeSceneId);

  if (activePoint && !isBaselineVigorScore(activePoint.score)) {
    setText("history-latest", "—");
    setText(
      "history-latest-label",
      activePoint.score === 0 ? "Zero-vigor observation excluded" : "No eligible score for this observation"
    );
    setText("history-change", "—");
    setText("history-change-label", "No comparable selected score");
    setBaselineSignal({
      status: "Not evaluated",
      message: activePoint.score === 0
        ? "An exact zero-vigor score is excluded from seasonal monitoring because it is likely non-crop evidence."
        : "This observation does not contain enough eligible crop evidence for seasonal monitoring.",
      tone: "signal-building",
    });
    return;
  }

  if (!scored.length) {
    setText("history-latest", "—");
    setText("history-latest-label", "No scored observations");
    setText("history-change", "—");
    setText("history-change-label", "Requires comparable scores");
    setBaselineSignal({
      status: "Baseline starting",
      message: "Clear crop evidence is needed before a seasonal baseline can form.",
      tone: "signal-building",
    });
    return;
  }

  const selected = activePoint && isBaselineVigorScore(activePoint.score)
    ? activePoint
    : scored[scored.length - 1];
  setText("history-latest", `${Math.round(selected.score)}/100`);
  setText(
    "history-latest-label",
    `${selected.label} · ${dateLabel(selected.date, { short: true, time: true })}`
  );

  // Reprocessed copies of one acquisition are not independent historical evidence.
  const earlierByAcquisition = new Map();
  scored
    .filter((point) => point.date.valueOf() < selected.date.valueOf())
    .sort((left, right) => left.date - right.date)
    .forEach((point) => earlierByAcquisition.set(point.date.valueOf(), point));
  const earlier = [...earlierByAcquisition.values()];
  const selectedSeason = seasonForDate(selected.date);
  const comparable = earlier.filter(
    (point) => seasonForDate(point.date) === selectedSeason
  );

  if (comparable.length) {
    const previous = comparable[comparable.length - 1];
    const change = selected.score - previous.score;
    setText("history-change", `${change >= 0 ? "+" : ""}${change.toFixed(1)} pts`);
    setText(
      "history-change-label",
      `Versus previous ${selectedSeason} · ${dateLabel(previous.date, { short: true })}`
    );
  } else {
    setText("history-change", "—");
    setText("history-change-label", `First eligible ${selectedSeason || "seasonal"} observation`);
  }

  setBaselineSignal(baselineSignalFor(selected, comparable, selectedSeason));
}

function setBaselineSignal(signal) {
  setText("baseline-status", signal.status);
  setText("baseline-message", signal.message);
  elements["baseline-card"].className = `baseline-card ${signal.tone}`;
}

function baselineSignalFor(selected, priorComparable, season) {
  const seasonLabel = season || "seasonal";
  const severity = slug(selected.label);
  const absoluteConditionSignal = (baselineState) => {
    if (severity === "high-anomaly") {
      return {
        status: baselineState === "starting" ? "Low condition · no baseline" : "Persistently low condition",
        message: baselineState === "starting"
          ? `The selected score is low, but this is the first eligible ${seasonLabel} observation in the timeline. Confirm conditions before action.`
          : `The selected condition remains low without an unusual change from the available ${seasonLabel} comparison. Confirm conditions in the field.`,
        tone: "signal-review",
      };
    }
    if (["moderate-anomaly", "watch"].includes(severity)) {
      return {
        status: "Condition watch",
        message: baselineState === "starting"
          ? `The selected score is below nominal, but no earlier ${seasonLabel} observation is available for a trend comparison.`
          : `The selected condition is below nominal and remains close to the available ${seasonLabel} comparison. Continue monitoring.`,
        tone: "signal-watch",
      };
    }
    return null;
  };

  if (!priorComparable.length) {
    return absoluteConditionSignal("starting") || {
      status: "Baseline starting",
      message: `This is the first eligible ${seasonLabel} observation in the timeline. A later ${seasonLabel} image can establish a comparison.`,
      tone: "signal-building",
    };
  }

  const previousScores = priorComparable.map((point) => point.score);
  const average = previousScores.reduce((total, value) => total + value, 0) / previousScores.length;
  const deviation = Math.sqrt(
    previousScores.reduce((total, value) => total + ((value - average) ** 2), 0) / previousScores.length
  );
  const difference = selected.score - average;
  const reviewMargin = Math.max(8, deviation * 2);
  const watchMargin = Math.max(4, deviation);
  const comparisonStrength = priorComparable.length === 1 ? "early" : "established";

  if (difference <= -reviewMargin) {
    return {
      status: comparisonStrength === "early" ? "Early review signal" : "Review signal",
      message: `The selected score is ${Math.abs(difference).toFixed(1)} points below the prior ${seasonLabel} mean. Confirm the change in the field.`,
      tone: "signal-review",
    };
  }
  if (difference <= -watchMargin) {
    return {
      status: comparisonStrength === "early" ? "Early watch signal" : "Watch signal",
      message: `The selected score is ${Math.abs(difference).toFixed(1)} points below the prior ${seasonLabel} mean. Increase monitoring frequency.`,
      tone: "signal-watch",
    };
  }
  if (difference >= reviewMargin) {
    return {
      status: priorComparable.length === 1 ? "Early improvement" : "Strong improvement",
      message: `The selected score is ${difference.toFixed(1)} points above the prior ${seasonLabel} mean. This is an encouraging relative signal.`,
      tone: "signal-improving",
    };
  }
  if (difference >= watchMargin) {
    return {
      status: "Improving signal",
      message: `The selected score is ${difference.toFixed(1)} points above the prior ${seasonLabel} mean. Continue monitoring to confirm the trend.`,
      tone: "signal-improving",
    };
  }

  return absoluteConditionSignal("available") || {
    status: priorComparable.length === 1 ? "Preliminary range" : "Within seasonal range",
    message: `The selected score remains close to the prior ${seasonLabel} observations; no unusual change is visible.`,
    tone: "signal-stable",
  };
}

function drawHistory(points) {
  const container = elements["history-chart"];
  container.replaceChildren();
  const allScored = points.filter(
    (point) => isBaselineVigorScore(point.score) && !Number.isNaN(point.date.valueOf())
  );
  if (!allScored.length) {
    container.append(historyMessage("No comparable scored observations are available yet."));
    return;
  }

  const scored = allScored.slice(-12);
  const width = 430;
  const rowGap = 64;
  const padding = { left: 98, right: 30, top: 40, bottom: 24 };
  const height = Math.max(286, padding.top + padding.bottom + ((scored.length - 1) * rowGap));
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `Crop-condition trajectory from earlier to later observations, scored from zero to one hundred${allScored.length > scored.length ? "; latest twelve shown" : ""}`
  );

  const defs = document.createElementNS(SVG_NS, "defs");
  const gradient = document.createElementNS(SVG_NS, "linearGradient");
  gradient.id = "historyArea";
  gradient.setAttribute("x1", "0");
  gradient.setAttribute("y1", "0");
  gradient.setAttribute("x2", "1");
  gradient.setAttribute("y2", "0");
  [["0%", "#58ad7c", "0.32"], ["100%", "#58ad7c", "0"]].forEach(([offset, color, opacity]) => {
    const stop = document.createElementNS(SVG_NS, "stop");
    stop.setAttribute("offset", offset);
    stop.setAttribute("stop-color", color);
    stop.setAttribute("stop-opacity", opacity);
    gradient.append(stop);
  });
  defs.append(gradient);
  svg.append(defs);

  const xAt = (score) => padding.left + (clamp(score, 0, 100) / 100) * plotWidth;
  const yAt = (index) => padding.top + (scored.length === 1
    ? plotHeight / 2
    : (index / (scored.length - 1)) * plotHeight);

  [0, 25, 50, 75, 100].forEach((score) => {
    const x = xAt(score);
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", padding.top - 10);
    line.setAttribute("y2", height - padding.bottom + 10);
    line.setAttribute("class", "chart-grid");
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", 15);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "chart-label");
    label.textContent = score;
    svg.append(line, label);
  });

  const pathData = scored
    .map((point, index) => `${index ? "L" : "M"}${xAt(point.score)},${yAt(index)}`)
    .join(" ");
  if (scored.length > 1) {
    const area = document.createElementNS(SVG_NS, "path");
    area.setAttribute(
      "d",
      `${pathData} L${padding.left},${yAt(scored.length - 1)} L${padding.left},${yAt(0)} Z`
    );
    area.setAttribute("class", "chart-area");
    const line = document.createElementNS(SVG_NS, "path");
    line.setAttribute("d", pathData);
    line.setAttribute("class", "chart-line");
    svg.append(area, line);
  }

  scored.forEach((point, index) => {
    const group = document.createElementNS(SVG_NS, "g");
    const isActive = point.scene.scene_id === state.activeSceneId;
    group.setAttribute("class", `chart-point-button${isActive ? " active" : ""}`);
    group.setAttribute("role", "button");
    group.setAttribute("tabindex", "0");
    group.setAttribute(
      "aria-label",
      `Load ${dateLabel(point.date, { time: true })} observation, score ${point.score.toFixed(1)}`
    );

    const hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("x", "1");
    hit.setAttribute("y", yAt(index) - 21);
    hit.setAttribute("width", width - 2);
    hit.setAttribute("height", "42");
    hit.setAttribute("rx", "8");
    hit.setAttribute("class", "chart-row-hit");

    const date = document.createElementNS(SVG_NS, "text");
    date.setAttribute("x", padding.left - 12);
    date.setAttribute("y", yAt(index) + 4);
    date.setAttribute("text-anchor", "end");
    date.setAttribute("class", "chart-date-label");
    date.textContent = dateLabel(point.date, { short: true });

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", xAt(point.score));
    circle.setAttribute("cy", yAt(index));
    circle.setAttribute("r", isActive ? "7" : "5");
    circle.setAttribute("class", `chart-point${isActive ? " active" : ""}`);
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${dateLabel(point.date, { time: true })} · ${point.score.toFixed(1)}/100 · ${point.label}`;
    circle.append(title);

    const score = document.createElementNS(SVG_NS, "text");
    const placeBefore = point.score >= 88;
    score.setAttribute("x", xAt(point.score) + (placeBefore ? -11 : 11));
    score.setAttribute("y", yAt(index) + 4);
    score.setAttribute("text-anchor", placeBefore ? "end" : "start");
    score.setAttribute("class", "chart-score-label");
    score.textContent = Math.round(point.score);

    const activate = () => selectScene(point.scene.scene_id);
    group.addEventListener("click", activate);
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
    group.append(hit, date, circle, score);
    svg.append(group);
  });
  container.append(svg);
}

function renderHistoryStrip(history, pointByScene) {
  const strip = elements["history-strip"];
  strip.replaceChildren();
  history.forEach((scene) => {
    const point = pointByScene.get(scene.scene_id);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item${scene.scene_id === state.activeSceneId ? " active" : ""}`;
    button.setAttribute("aria-current", scene.scene_id === state.activeSceneId ? "date" : "false");
    const image = document.createElement("img");
    image.className = "history-thumb";
    image.src = scene.links.preview;
    image.alt = "";
    image.loading = "lazy";
    const copy = document.createElement("span");
    const date = document.createElement("span");
    date.textContent = dateLabel(scene.acquired_at, { short: true, time: true });
    const label = document.createElement("small");
    const rawScore = point ? point.score : scene.condition?.score;
    label.textContent = rawScore === 0
      ? "Excluded zero-vigor score"
      : point?.label || scene.condition?.label || "No score";
    copy.append(date, label);
    const score = document.createElement("strong");
    const value = rawScore;
    score.textContent = isNumber(value) ? Math.round(value) : "—";
    score.style.color = conditionColor(value);
    button.append(image, copy, score);
    button.addEventListener("click", () => selectScene(scene.scene_id));
    strip.append(button);
  });
  window.requestAnimationFrame(() => {
    const active = strip.querySelector(".history-item.active");
    if (active) {
      strip.scrollLeft = Math.max(0, active.offsetLeft - (strip.clientWidth - active.clientWidth) / 2);
    }
  });
}

function setLayer(layer) {
  document.querySelectorAll(".layer-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.layer === layer);
  });
  elements["viewer-transform"].classList.remove("layer-satellite", "layer-health", "layer-both");
  elements["viewer-transform"].classList.add(`layer-${layer}`);
}

async function loadPipelineCapability() {
  try {
    state.pipelineCapability = await apiFetch(`${API}/pipeline-runs/capability`);
    const capability = state.pipelineCapability;
    elements["analysis-capability"].classList.toggle("unavailable", !capability.available);
    setText(
      "analysis-capability",
      capability.available
        ? "Ready to run one observation at a time through the current accelerated payload pipeline."
        : capability.reason
    );
    elements["start-analysis"].disabled = !capability.available;
  } catch (error) {
    state.pipelineCapability = { available: false, reason: error.message };
    elements["analysis-capability"].classList.add("unavailable");
    setText("analysis-capability", error.message);
    elements["start-analysis"].disabled = true;
  }
}

async function openAnalysisDialog() {
  if (!state.pipelineCapability) await loadPipelineCapability();
  if (state.pipelineRun && ["queued", "running"].includes(state.pipelineRun.status)) {
    showRunPanel();
  } else {
    showAnalysisFields();
    elements["analysis-region"].value = state.activeRegionId || "demo-region";
    if (state.manifest?.sensor) elements["analysis-sensor"].value = state.manifest.sensor;
    updateImageField();
  }
  if (!elements["analysis-dialog"].open) elements["analysis-dialog"].showModal();
}

function showAnalysisFields() {
  elements["analysis-fields"].classList.remove("hidden");
  elements["analysis-progress-panel"].classList.add("hidden");
}

function showRunPanel() {
  elements["analysis-fields"].classList.add("hidden");
  elements["analysis-progress-panel"].classList.remove("hidden");
  renderPipelineRun();
}

function updateImageField() {
  const sentinel = elements["analysis-sensor"].value === "sentinel-2";
  elements["image-field"].classList.toggle("hidden", !sentinel);
  if (!sentinel) elements["analysis-image"].value = "";
}

async function startAnalysis(event) {
  event.preventDefault();
  if (!elements["analysis-form"].reportValidity()) return;
  const payload = {
    sensor: elements["analysis-sensor"].value,
    region_id: elements["analysis-region"].value.trim(),
  };
  const inputPath = elements["analysis-input"].value.trim();
  const image = elements["analysis-image"].value.trim();
  if (inputPath) payload.input_path = inputPath;
  if (image && payload.sensor === "sentinel-2") payload.image = image;
  elements["start-analysis"].disabled = true;
  try {
    const response = await apiFetch(`${API}/pipeline-runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.pipelineRun = response.run;
    state.resultLoadedForRun = null;
    showRunPanel();
    startRunMonitoring();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements["start-analysis"].disabled = !state.pipelineCapability?.available;
  }
}

function startRunMonitoring() {
  window.clearInterval(state.runPollTimer);
  window.clearInterval(state.runClockTimer);
  state.runPollTimer = window.setInterval(pollPipelineRun, 1000);
  state.runClockTimer = window.setInterval(renderRunClock, 250);
  renderPipelineRun();
}

async function pollPipelineRun() {
  try {
    const response = await apiFetch(`${API}/pipeline-runs/current`);
    if (!response.run) return;
    state.pipelineRun = response.run;
    renderPipelineRun();
    if (["succeeded", "failed"].includes(response.run.status)) {
      window.clearInterval(state.runPollTimer);
      window.clearInterval(state.runClockTimer);
      if (response.run.status === "succeeded" && state.resultLoadedForRun !== response.run.run_id) {
        state.resultLoadedForRun = response.run.run_id;
        await loadScenes({ targetSceneId: response.run.scene_id, preserveSelection: false });
      }
    }
  } catch (error) {
    window.clearInterval(state.runPollTimer);
    showToast(error.message, true);
  }
}

function renderRunClock() {
  const run = state.pipelineRun;
  if (!run) return;
  let seconds = run.elapsed_seconds;
  if (["queued", "running"].includes(run.status)) {
    const started = new Date(run.started_at || run.created_at);
    seconds = Number.isNaN(started.valueOf()) ? 0 : (Date.now() - started.valueOf()) / 1000;
  }
  setText("run-elapsed", formatDuration(seconds));
}

function formatDuration(seconds) {
  if (!isNumber(seconds)) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

function renderPipelineRun() {
  const run = state.pipelineRun;
  if (!run) return;
  renderRunClock();
  setText("run-payload-time", isNumber(run.payload_seconds) ? `${run.payload_seconds.toFixed(2)} s` : "—");
  elements["view-result"].classList.add("hidden");
  elements["retry-analysis"].classList.add("hidden");
  elements["run-visual"].className = "run-visual";
  if (run.status === "succeeded") {
    elements["run-visual"].classList.add("complete");
    setText("run-status-kicker", "Analysis complete");
    setText("run-status-title", "Observation ready");
    setText("run-status-message", run.message);
    elements["view-result"].classList.remove("hidden");
  } else if (run.status === "failed") {
    elements["run-visual"].classList.add("failed");
    setText("run-status-kicker", "Analysis stopped");
    setText("run-status-title", "Could not complete analysis");
    setText("run-status-message", run.message);
    elements["retry-analysis"].classList.remove("hidden");
  } else {
    setText("run-status-kicker", "Payload analysis");
    setText("run-status-title", run.status === "queued" ? "Preparing the analysis…" : "Analyzing the observation…");
    setText("run-status-message", "The dashboard will update automatically when the observation is ready.");
  }
}

function viewRunResult() {
  elements["analysis-dialog"].close();
  if (state.pipelineRun?.scene_id && state.activeSceneId !== state.pipelineRun.scene_id) {
    loadScenes({ targetSceneId: state.pipelineRun.scene_id, preserveSelection: false });
  }
}

function retryAnalysis() {
  state.pipelineRun = null;
  showAnalysisFields();
}

elements["region-select"].addEventListener("change", (event) => selectRegion(event.target.value));
elements["theme-toggle"].addEventListener("click", toggleTheme);
elements["refresh-scenes"].addEventListener("click", () => loadScenes());
elements["new-analysis"].addEventListener("click", openAnalysisDialog);
elements["empty-new-analysis"].addEventListener("click", openAnalysisDialog);
elements["close-analysis"].addEventListener("click", () => elements["analysis-dialog"].close());
elements["analysis-sensor"].addEventListener("change", updateImageField);
elements["analysis-form"].addEventListener("submit", startAnalysis);
elements["view-result"].addEventListener("click", viewRunResult);
elements["retry-analysis"].addEventListener("click", retryAnalysis);
elements["draw-area"].addEventListener("click", () => setDrawMode(!state.drawMode));
elements["clear-area"].addEventListener("click", clearAreaSelection);
document.querySelectorAll(".layer-button").forEach((button) => {
  button.addEventListener("click", () => setLayer(button.dataset.layer));
});
elements["zoom-in"].addEventListener("click", () => zoomViewer(1.35));
elements["zoom-out"].addEventListener("click", () => zoomViewer(1 / 1.35));
elements["reset-view"].addEventListener("click", resetViewer);

elements["scene-viewer"].addEventListener("wheel", (event) => {
  event.preventDefault();
  zoomViewer(event.deltaY < 0 ? 1.18 : 1 / 1.18, event.clientX, event.clientY);
}, { passive: false });

elements["scene-viewer"].addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || event.target.closest("button")) return;
  const point = imagePointFromEvent(event);
  if (state.drawMode && point) {
    state.interaction = "draw";
    state.interactionStart = point;
    state.draftPixels = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
    renderSelectionBox(state.draftPixels);
  } else if (!state.drawMode) {
    state.interaction = "pan";
    state.interactionStart = { x: event.clientX, y: event.clientY };
    state.panStart = { x: state.panX, y: state.panY };
    elements["scene-viewer"].classList.add("dragging");
  } else {
    return;
  }
  state.pointerId = event.pointerId;
  elements["scene-viewer"].setPointerCapture(event.pointerId);
});

elements["scene-viewer"].addEventListener("pointermove", (event) => {
  updateCursorCoordinate(event);
  if (event.pointerId !== state.pointerId) return;
  if (state.interaction === "pan") {
    state.panX = state.panStart.x + event.clientX - state.interactionStart.x;
    state.panY = state.panStart.y + event.clientY - state.interactionStart.y;
    applyViewerTransform();
  } else if (state.interaction === "draw") {
    const point = imagePointFromEvent(event);
    if (!point) return;
    state.draftPixels = {
      x1: Math.min(state.interactionStart.x, point.x),
      y1: Math.min(state.interactionStart.y, point.y),
      x2: Math.max(state.interactionStart.x, point.x),
      y2: Math.max(state.interactionStart.y, point.y),
    };
    renderSelectionBox(state.draftPixels);
  }
});

function endPointerInteraction(event) {
  if (event.pointerId !== state.pointerId) return;
  const completedInteraction = state.interaction;
  const completedDraft = state.draftPixels;
  state.pointerId = null;
  state.interaction = null;
  state.interactionStart = null;
  elements["scene-viewer"].classList.remove("dragging");
  if (completedInteraction === "draw") finishAreaSelection(completedDraft);
}

elements["scene-viewer"].addEventListener("pointerup", endPointerInteraction);
elements["scene-viewer"].addEventListener("pointercancel", endPointerInteraction);
elements["scene-viewer"].addEventListener("pointerleave", () => setText("cursor-coordinate", "—"));
elements["scene-viewer"].addEventListener("keydown", (event) => {
  if (event.key === "+" || event.key === "=") zoomViewer(1.25);
  if (event.key === "-") zoomViewer(0.8);
  if (event.key === "0") resetViewer();
  if (event.key.toLowerCase() === "s") setDrawMode(!state.drawMode);
  if (event.key === "Escape" && state.drawMode) setDrawMode(false);
});

window.addEventListener("resize", () => {
  window.clearTimeout(window.vitaResizeTimer);
  window.vitaResizeTimer = window.setTimeout(resetViewer, 120);
});

setLayer("both");
applyTheme(currentTheme());
const colorPreference = window.matchMedia("(prefers-color-scheme: dark)");
colorPreference.addEventListener("change", (event) => {
  if (!storedTheme()) applyTheme(event.matches ? "dark" : "light");
});
Promise.allSettled([
  checkService(),
  loadPipelineCapability(),
  loadScenes({ preserveSelection: false }),
]);
