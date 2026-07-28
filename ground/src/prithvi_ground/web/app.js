"use strict";

const API = "/api/v1";
const NS = "http://www.w3.org/2000/svg";

const elements = Object.fromEntries(
  [
    "service-indicator", "service-label", "scene-search", "refresh-scenes", "scene-list",
    "scene-title", "scene-subtitle", "sensor-badge", "manifest-link", "empty-state",
    "dashboard", "score-ring", "condition-score", "condition-label", "score-explanation",
    "score-details-toggle", "score-methodology", "evidence-score", "evidence-label",
    "evidence-progress", "analysis-coverage", "analysis-pixels", "analysis-progress",
    "usable-coverage", "unusable-caption", "usable-progress", "absolute-vigor",
    "spatial-penalty", "low-vigor-share", "relative-anomaly-share", "weight-description",
    "scene-viewer", "viewer-transform", "scene-image", "condition-overlay", "grid-overlay",
    "zoom-out", "zoom-in", "reset-view", "overlay-toggle", "overlay-opacity",
    "opacity-value", "grid-toggle", "cursor-coordinate", "legend", "cell-title",
    "cell-score-badge", "cell-bounds", "cell-ndvi", "cell-gndvi", "cell-evi", "cell-savi",
    "cell-analysis", "cell-alert", "cell-unusable", "cell-crop", "metric-grid", "cloud-bar",
    "quality-list", "history-count", "history-chart", "explanation-list", "scientific-claim",
    "toast",
  ].map((id) => [id, document.getElementById(id)])
);

const state = {
  scenes: [],
  activeSceneId: null,
  manifest: null,
  selectedCellId: null,
  imageWidth: 1,
  imageHeight: 1,
  baseScale: 1,
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragMoved: false,
  pointerId: null,
  dragStartX: 0,
  dragStartY: 0,
  panStartX: 0,
  panStartY: 0,
};

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function number(value, digits = 2) {
  return isNumber(value) ? value.toFixed(digits) : "—";
}

function percentage(value, digits = 1) {
  return isNumber(value) ? `${value.toFixed(digits)}%` : "—";
}

function integer(value) {
  return isNumber(value) ? Math.round(value).toLocaleString() : "—";
}

function slug(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function humanizeIdentifier(value) {
  const words = String(value || "Unknown region").replace(/[_-]+/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Unknown region";
}

function dateLabel(value, includeTime = false) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return String(value || "Unknown date");
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    ...(includeTime ? { hour: "2-digit", minute: "2-digit", timeZoneName: "short" } : {}),
  }).format(date);
}

function conditionColor(score) {
  if (!isNumber(score)) return "#61776d";
  if (score >= 75) return "#67d391";
  if (score >= 55) return "#f1bd62";
  if (score >= 35) return "#ed8b54";
  return "#e55e5e";
}

function setText(id, value) {
  elements[id].textContent = value;
}

function setProgress(id, value) {
  elements[id].style.width = `${Math.max(0, Math.min(100, isNumber(value) ? value : 0))}%`;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => elements.toast.classList.remove("show"), 3600);
}

async function apiFetch(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_error) {
      // Keep the HTTP status fallback when a response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
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

async function loadScenes({ preserveSelection = true } = {}) {
  elements["scene-list"].replaceChildren(messageNode("Loading verified observations…"));
  try {
    const response = await apiFetch(`${API}/scenes?limit=500`);
    state.scenes = response.items;
    renderSceneList(elements["scene-search"].value);
    if (!state.scenes.length) {
      showEmptyState();
      return;
    }
    const preserved = preserveSelection && state.scenes.some((item) => item.scene_id === state.activeSceneId);
    await selectScene(preserved ? state.activeSceneId : state.scenes[0].scene_id);
  } catch (error) {
    state.scenes = [];
    elements["scene-list"].replaceChildren(messageNode(error.message));
    showEmptyState();
    showToast(error.message, true);
  }
}

function messageNode(message) {
  const node = document.createElement("p");
  node.className = "list-message";
  node.textContent = message;
  return node;
}

function renderSceneList(query = "") {
  const normalized = query.trim().toLowerCase();
  const scenes = state.scenes.filter((scene) =>
    [scene.scene_id, scene.region_id, scene.sensor, scene.condition.label]
      .join(" ")
      .toLowerCase()
      .includes(normalized)
  );
  elements["scene-list"].replaceChildren();
  if (!scenes.length) {
    elements["scene-list"].append(messageNode(normalized ? "No matching observation." : "No observations received yet."));
    return;
  }
  scenes.forEach((scene) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `scene-item ${slug(scene.condition.label)}`;
    button.classList.toggle("active", scene.scene_id === state.activeSceneId);
    button.setAttribute("aria-current", scene.scene_id === state.activeSceneId ? "true" : "false");
    const marker = document.createElement("i");
    marker.setAttribute("aria-hidden", "true");
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = humanizeIdentifier(scene.region_id);
    const condition = document.createElement("span");
    condition.textContent = `${scene.condition.label} · ${isNumber(scene.condition.score) ? Math.round(scene.condition.score) : "—"}/100`;
    const meta = document.createElement("small");
    meta.textContent = `${dateLabel(scene.acquired_at)} · ${scene.sensor}`;
    copy.append(title, condition, meta);
    button.append(marker, copy);
    button.addEventListener("click", () => selectScene(scene.scene_id));
    elements["scene-list"].append(button);
  });
}

function showEmptyState() {
  elements.dashboard.classList.add("hidden");
  elements["empty-state"].classList.remove("hidden");
  setText("scene-title", "Crop-condition overview");
  setText("scene-subtitle", "Select a verified scene to begin.");
  setText("sensor-badge", "No scene");
  elements["manifest-link"].classList.add("disabled");
}

async function selectScene(sceneId) {
  if (!sceneId) return;
  state.activeSceneId = sceneId;
  state.selectedCellId = null;
  renderSceneList(elements["scene-search"].value);
  try {
    const response = await apiFetch(`${API}/scenes/${encodeURIComponent(sceneId)}/manifest`);
    if (state.activeSceneId !== sceneId) return;
    state.manifest = response.scene;
    renderManifest(response.scene, response.links);
    await renderHistory(response.scene.region_id);
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderManifest(manifest, links) {
  elements["empty-state"].classList.add("hidden");
  elements.dashboard.classList.remove("hidden");
  setText("scene-title", humanizeIdentifier(manifest.region_id));
  setText(
    "scene-subtitle",
    `${dateLabel(manifest.acquired_at, true)} · ${manifest.scene_id} · ${formatBounds(manifest.geospatial.bounds_wgs84)}`
  );
  setText("sensor-badge", manifest.sensor.replace("-", " "));
  elements["manifest-link"].href = links.manifest;
  elements["manifest-link"].classList.remove("disabled");

  renderCondition(manifest);
  renderMetrics(manifest.metrics || {});
  renderQuality(manifest.quality || {});
  renderLegend(manifest.legend || {});
  renderExplanations(manifest);
  prepareViewer(manifest, links);
  resetCellInspector();
}

function renderCondition(manifest) {
  const condition = manifest.condition || {};
  const quality = manifest.quality || {};
  const score = condition.condition_score;
  const scoreDisplay = isNumber(score) ? Math.round(score).toString() : "—";
  setText("condition-score", scoreDisplay);
  elements["score-ring"].style.setProperty("--score", isNumber(score) ? score : 0);
  elements["score-ring"].style.setProperty("--score-color", conditionColor(score));
  elements["score-ring"].setAttribute(
    "aria-label",
    isNumber(score) ? `Crop condition score ${score.toFixed(1)} out of 100` : "Crop condition score unavailable"
  );
  setText("condition-label", condition.label || "Insufficient data");
  elements["condition-label"].className = `condition-pill ${slug(condition.label)}`;
  setText(
    "score-explanation",
    isNumber(score)
      ? `Relative spectral vigor from ${integer(condition.analysis_pixels)} clear crop pixels; higher means stronger agreement with the configured prototype ranges.`
      : "There is not enough clear crop evidence for a responsible score."
  );

  setText("evidence-score", isNumber(condition.evidence_quality_score) ? `${Math.round(condition.evidence_quality_score)}/100` : "—");
  setText("evidence-label", condition.evidence_quality_label ? `${condition.evidence_quality_label} coverage evidence` : "Coverage indicator");
  setProgress("evidence-progress", condition.evidence_quality_score);
  setText("analysis-coverage", percentage(condition.analysis_percentage));
  setText("analysis-pixels", `${integer(condition.analysis_pixels)} clear crop pixels`);
  setProgress("analysis-progress", condition.analysis_percentage);
  setText("usable-coverage", percentage(quality.usable_percentage));
  setText("unusable-caption", `${percentage(quality.unusable_percentage)} masked as unusable`);
  setProgress("usable-progress", quality.usable_percentage);

  setText("absolute-vigor", isNumber(condition.absolute_vigor_score) ? `${condition.absolute_vigor_score.toFixed(1)}/100` : "—");
  setText("spatial-penalty", isNumber(condition.spatial_penalty) ? `−${condition.spatial_penalty.toFixed(1)}` : "—");
  setText("low-vigor-share", percentage(condition.low_vigor_percentage));
  setText("relative-anomaly-share", percentage(condition.relative_anomaly_percentage));
  const config = condition.configuration || {};
  const weights = [
    ["NDVI", config.ndvi_weight],
    ["GNDVI", config.gndvi_weight],
    ["EVI", config.evi_weight],
    ["SAVI", config.savi_weight],
  ].filter((item) => isNumber(item[1]));
  setText(
    "weight-description",
    weights.length
      ? `Configured index weights: ${weights.map(([name, weight]) => `${name} ${(weight * 100).toFixed(0)}%`).join(", ")}. The regional score combines robust median and lower-quartile pixel condition before any spatial penalty.`
      : "Index weights and decision thresholds are preserved in the scene manifest for auditability."
  );
}

function renderMetrics(metrics) {
  const definitions = [
    ["ndvi", "NDVI", "Canopy vigor", "#67d391"],
    ["gndvi", "GNDVI", "Green chlorophyll response", "#b9dd74"],
    ["evi", "EVI", "Blue-corrected canopy vigor", "#42bfd1"],
    ["savi", "SAVI", "Soil-adjusted vegetation", "#f1bd62"],
    ["cvi", "CVI", "Chlorophyll ratio · diagnostic", "#8e79d8"],
    ["vari", "VARI", "Visible greenness · diagnostic", "#52b788"],
    ["excess_green", "EXG", "RGB excess green · diagnostic", "#95d5b2"],
    ["rgb_brightness", "RGB", "Visible brightness · diagnostic", "#a7b6af"],
  ];
  elements["metric-grid"].replaceChildren();
  definitions.forEach(([key, label, description, color]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.style.setProperty("--metric-color", color);
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = number(metrics[key]?.median, 3);
    const caption = document.createElement("small");
    caption.textContent = description;
    card.append(name, value, caption);
    elements["metric-grid"].append(card);
  });
}

function renderQuality(quality) {
  const classes = [
    ["clear_percentage", "Clear", "#3aaa72"],
    ["thick_cloud_percentage", "Thick cloud", "#f5f7fa"],
    ["thin_cloud_percentage", "Thin cloud", "#00b8d9"],
    ["cloud_shadow_percentage", "Cloud shadow", "#7e57c2"],
    ["invalid_percentage", "Invalid", "#ffc107"],
  ];
  elements["cloud-bar"].replaceChildren();
  elements["quality-list"].replaceChildren();
  classes.forEach(([key, label, color]) => {
    const value = isNumber(quality[key]) ? Math.max(0, quality[key]) : 0;
    const segment = document.createElement("i");
    segment.className = "cloud-segment";
    segment.style.width = `${value}%`;
    segment.style.background = color;
    segment.title = `${label}: ${percentage(value)}`;
    elements["cloud-bar"].append(segment);

    const row = document.createElement("div");
    row.className = "quality-row";
    const name = document.createElement("span");
    const dot = document.createElement("i");
    dot.className = "quality-dot";
    dot.style.background = color;
    name.append(dot, document.createTextNode(label));
    const numberNode = document.createElement("strong");
    numberNode.textContent = percentage(quality[key]);
    row.append(name, numberNode);
    elements["quality-list"].append(row);
  });
}

function safeRgba(value) {
  if (!Array.isArray(value) || value.length !== 4 || value.some((item) => !isNumber(item))) return "rgba(97,119,109,.8)";
  const [red, green, blue, alpha] = value.map((item) => Math.max(0, Math.min(255, item)));
  return `rgba(${red}, ${green}, ${blue}, ${(alpha / 255).toFixed(3)})`;
}

function renderLegend(legend) {
  elements.legend.replaceChildren();
  const gradient = document.createElement("span");
  gradient.className = "legend-item";
  const gradientSwatch = document.createElement("i");
  gradientSwatch.className = "legend-gradient";
  gradient.append(gradientSwatch, document.createTextNode("Condition 0 → 100"));
  elements.legend.append(gradient);

  const labels = {
    thick_cloud: "Thick cloud",
    thin_cloud: "Thin cloud",
    cloud_shadow: "Cloud shadow",
    invalid: "Invalid",
    unusable_buffer: "Safety buffer",
  };
  Object.entries(labels).forEach(([key, label]) => {
    if (!legend.classes?.[key]) return;
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.className = "legend-swatch";
    swatch.style.background = safeRgba(legend.classes[key].rgba);
    item.append(swatch, document.createTextNode(label));
    elements.legend.append(item);
  });
}

function renderExplanations(manifest) {
  elements["explanation-list"].replaceChildren();
  const explanations = manifest.condition?.explanations || [];
  (explanations.length ? explanations : ["No interpretation was supplied for this observation."]).forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    elements["explanation-list"].append(item);
  });
  setText("scientific-claim", manifest.claim || "Relative crop-condition screening; not an agronomic diagnosis.");
}

function prepareViewer(manifest, links) {
  state.imageWidth = manifest.assets.rgb_preview.width;
  state.imageHeight = manifest.assets.rgb_preview.height;
  elements["viewer-transform"].style.width = `${state.imageWidth}px`;
  elements["viewer-transform"].style.height = `${state.imageHeight}px`;
  elements["scene-image"].src = links.preview;
  elements["condition-overlay"].src = links.condition_overlay;
  elements["condition-overlay"].style.opacity = elements["overlay-toggle"].checked
    ? (Number(elements["overlay-opacity"].value) / 100).toString()
    : "0";
  renderGrid(manifest.interaction_grid);
  const informativeCell = [...(manifest.interaction_grid?.cells || [])]
    .filter((cell) => isNumber(cell.condition_score))
    .sort((left, right) => (right.quality?.analysis_pixels || 0) - (left.quality?.analysis_pixels || 0))[0];
  if (informativeCell) selectCell(informativeCell.id);
  window.requestAnimationFrame(resetViewer);
}

function renderGrid(grid) {
  elements["grid-overlay"].replaceChildren();
  if (!grid || !Number.isInteger(grid.rows) || !Number.isInteger(grid.columns)) return;
  grid.cells.forEach((cell) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "grid-cell";
    button.classList.toggle("no-data", !isNumber(cell.condition_score));
    button.style.left = `${(cell.column / grid.columns) * 100}%`;
    button.style.top = `${(cell.row / grid.rows) * 100}%`;
    button.style.width = `${100 / grid.columns}%`;
    button.style.height = `${100 / grid.rows}%`;
    button.dataset.cellId = cell.id;
    button.setAttribute(
      "aria-label",
      `${cell.id}, condition ${isNumber(cell.condition_score) ? cell.condition_score.toFixed(1) : "insufficient data"}`
    );
    button.title = `${cell.id} · ${isNumber(cell.condition_score) ? `${cell.condition_score.toFixed(1)}/100` : "Insufficient data"}`;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectCell(cell.id);
    });
    elements["grid-overlay"].append(button);
  });
}

async function selectCell(cellId) {
  if (!state.activeSceneId) return;
  state.selectedCellId = cellId;
  elements["grid-overlay"].querySelectorAll(".grid-cell").forEach((cell) => {
    cell.classList.toggle("selected", cell.dataset.cellId === cellId);
  });
  try {
    const response = await apiFetch(
      `${API}/scenes/${encodeURIComponent(state.activeSceneId)}/cells/${encodeURIComponent(cellId)}`
    );
    if (state.selectedCellId === cellId) renderCell(response.cell);
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderCell(cell) {
  const metrics = cell.metrics || {};
  const quality = cell.quality || {};
  setText("cell-title", `Grid cell ${cell.id}`);
  setText("cell-score-badge", isNumber(cell.condition_score) ? Math.round(cell.condition_score).toString() : "—");
  elements["cell-score-badge"].style.color = conditionColor(cell.condition_score);
  setText("cell-bounds", formatBounds(cell.bounds_wgs84));
  setText("cell-ndvi", number(metrics.ndvi, 3));
  setText("cell-gndvi", number(metrics.gndvi, 3));
  setText("cell-evi", number(metrics.evi, 3));
  setText("cell-savi", number(metrics.savi, 3));
  setText("cell-analysis", percentage(quality.analysis_percentage));
  setText("cell-alert", percentage(cell.alert_percentage));
  setText("cell-unusable", percentage(quality.unusable_percentage));
  setText("cell-crop", percentage(quality.crop_candidate_percentage));
}

function resetCellInspector() {
  setText("cell-title", "Select a cell");
  setText("cell-score-badge", "—");
  elements["cell-score-badge"].style.color = "";
  setText("cell-bounds", "Click an outlined area on the scene for localized measurements.");
  ["cell-ndvi", "cell-gndvi", "cell-evi", "cell-savi", "cell-analysis", "cell-alert", "cell-unusable", "cell-crop"].forEach((id) => setText(id, "—"));
}

function formatBounds(bounds) {
  if (!Array.isArray(bounds) || bounds.length !== 4 || bounds.some((item) => !isNumber(item))) return "Coordinates unavailable";
  return `${bounds[1].toFixed(5)}, ${bounds[0].toFixed(5)} → ${bounds[3].toFixed(5)}, ${bounds[2].toFixed(5)}`;
}

function resetViewer() {
  const rect = elements["scene-viewer"].getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  state.baseScale = Math.min(
    Math.max(0.01, (rect.width - 32) / state.imageWidth),
    Math.max(0.01, (rect.height - 32) / state.imageHeight)
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
}

function zoomViewer(factor, clientX = null, clientY = null) {
  const oldZoom = state.zoom;
  const newZoom = Math.max(1, Math.min(12, oldZoom * factor));
  if (newZoom === oldZoom) return;
  const rect = elements["scene-viewer"].getBoundingClientRect();
  const pointerX = clientX === null ? rect.width / 2 : clientX - rect.left;
  const pointerY = clientY === null ? rect.height / 2 : clientY - rect.top;
  const offsetX = pointerX - rect.width / 2 - state.panX;
  const offsetY = pointerY - rect.height / 2 - state.panY;
  const ratio = newZoom / oldZoom;
  state.panX += offsetX * (1 - ratio);
  state.panY += offsetY * (1 - ratio);
  state.zoom = newZoom;
  applyViewerTransform();
}

function updateCursorCoordinate(event) {
  const bounds = state.manifest?.geospatial?.bounds_wgs84;
  if (!Array.isArray(bounds) || bounds.length !== 4) return;
  const rect = elements["scene-viewer"].getBoundingClientRect();
  const scale = state.baseScale * state.zoom;
  const imageX = (event.clientX - rect.left - rect.width / 2 - state.panX) / scale + state.imageWidth / 2;
  const imageY = (event.clientY - rect.top - rect.height / 2 - state.panY) / scale + state.imageHeight / 2;
  if (imageX < 0 || imageY < 0 || imageX > state.imageWidth || imageY > state.imageHeight) {
    setText("cursor-coordinate", "Outside scene");
    return;
  }
  const longitude = bounds[0] + (imageX / state.imageWidth) * (bounds[2] - bounds[0]);
  const latitude = bounds[3] - (imageY / state.imageHeight) * (bounds[3] - bounds[1]);
  setText("cursor-coordinate", `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
}

async function renderHistory(regionId) {
  try {
    const response = await apiFetch(`${API}/regions/${encodeURIComponent(regionId)}/history`);
    setText("history-count", `${response.count} observation${response.count === 1 ? "" : "s"}`);
    drawHistory(response.items);
  } catch (_error) {
    setText("history-count", "Unavailable");
    elements["history-chart"].replaceChildren(messageNode("No compatible history is available."));
  }
}

function drawHistory(items) {
  const points = items
    .filter((item) => isNumber(item.condition.score))
    .map((item) => ({ date: new Date(item.acquired_at), score: item.condition.score, label: item.condition.label }))
    .filter((point) => !Number.isNaN(point.date.valueOf()));
  elements["history-chart"].replaceChildren();
  if (!points.length) {
    const empty = messageNode("No scored observations yet.");
    empty.className = "history-empty";
    elements["history-chart"].append(empty);
    return;
  }
  const width = 640;
  const height = 172;
  const padding = { left: 34, right: 14, top: 15, bottom: 29 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Condition score history from zero to one hundred");

  const defs = document.createElementNS(NS, "defs");
  const gradient = document.createElementNS(NS, "linearGradient");
  gradient.id = "historyArea";
  gradient.setAttribute("x1", "0");
  gradient.setAttribute("y1", "0");
  gradient.setAttribute("x2", "0");
  gradient.setAttribute("y2", "1");
  [["0%", "#67d391", "0.65"], ["100%", "#67d391", "0"]].forEach(([offset, color, opacity]) => {
    const stop = document.createElementNS(NS, "stop");
    stop.setAttribute("offset", offset);
    stop.setAttribute("stop-color", color);
    stop.setAttribute("stop-opacity", opacity);
    gradient.append(stop);
  });
  defs.append(gradient);
  svg.append(defs);

  [0, 25, 50, 75, 100].forEach((score) => {
    const y = padding.top + plotHeight - (score / 100) * plotHeight;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("class", "chart-grid");
    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", padding.left - 8);
    label.setAttribute("y", y + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "chart-label");
    label.textContent = score;
    svg.append(line, label);
  });

  const xAt = (index) => padding.left + (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth);
  const yAt = (score) => padding.top + plotHeight - (score / 100) * plotHeight;
  const pathData = points.map((point, index) => `${index ? "L" : "M"}${xAt(index)},${yAt(point.score)}`).join(" ");
  if (points.length > 1) {
    const area = document.createElementNS(NS, "path");
    area.setAttribute("d", `${pathData} L${xAt(points.length - 1)},${padding.top + plotHeight} L${xAt(0)},${padding.top + plotHeight} Z`);
    area.setAttribute("class", "chart-area");
    const line = document.createElementNS(NS, "path");
    line.setAttribute("d", pathData);
    line.setAttribute("class", "chart-line");
    svg.append(area, line);
  }
  points.forEach((point, index) => {
    const circle = document.createElementNS(NS, "circle");
    circle.setAttribute("cx", xAt(index));
    circle.setAttribute("cy", yAt(point.score));
    circle.setAttribute("r", "4");
    circle.setAttribute("class", "chart-point");
    const title = document.createElementNS(NS, "title");
    title.textContent = `${dateLabel(point.date.toISOString())}: ${point.score.toFixed(1)} (${point.label})`;
    circle.append(title);
    svg.append(circle);
  });
  const firstDate = document.createElementNS(NS, "text");
  firstDate.setAttribute("x", padding.left);
  firstDate.setAttribute("y", height - 7);
  firstDate.setAttribute("class", "chart-label");
  firstDate.textContent = dateLabel(points[0].date.toISOString());
  const lastDate = document.createElementNS(NS, "text");
  lastDate.setAttribute("x", width - padding.right);
  lastDate.setAttribute("y", height - 7);
  lastDate.setAttribute("text-anchor", "end");
  lastDate.setAttribute("class", "chart-label");
  lastDate.textContent = dateLabel(points[points.length - 1].date.toISOString());
  svg.append(firstDate, lastDate);
  elements["history-chart"].append(svg);
}

elements["scene-search"].addEventListener("input", (event) => renderSceneList(event.target.value));
elements["refresh-scenes"].addEventListener("click", () => loadScenes());
elements["score-details-toggle"].addEventListener("click", () => {
  const willOpen = elements["score-methodology"].classList.contains("hidden");
  elements["score-methodology"].classList.toggle("hidden", !willOpen);
  elements["score-details-toggle"].setAttribute("aria-expanded", willOpen ? "true" : "false");
});
elements["overlay-toggle"].addEventListener("change", () => {
  elements["condition-overlay"].style.opacity = elements["overlay-toggle"].checked
    ? (Number(elements["overlay-opacity"].value) / 100).toString()
    : "0";
});
elements["overlay-opacity"].addEventListener("input", (event) => {
  setText("opacity-value", `${event.target.value}%`);
  if (elements["overlay-toggle"].checked) {
    elements["condition-overlay"].style.opacity = (Number(event.target.value) / 100).toString();
  }
});
elements["grid-toggle"].addEventListener("change", () => {
  elements["grid-overlay"].classList.toggle("off", !elements["grid-toggle"].checked);
});
elements["zoom-in"].addEventListener("click", () => zoomViewer(1.35));
elements["zoom-out"].addEventListener("click", () => zoomViewer(1 / 1.35));
elements["reset-view"].addEventListener("click", resetViewer);
elements["scene-viewer"].addEventListener("wheel", (event) => {
  event.preventDefault();
  zoomViewer(event.deltaY < 0 ? 1.18 : 1 / 1.18, event.clientX, event.clientY);
}, { passive: false });
elements["scene-viewer"].addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || event.target.closest(".grid-cell")) return;
  state.dragging = true;
  state.dragMoved = false;
  state.pointerId = event.pointerId;
  state.dragStartX = event.clientX;
  state.dragStartY = event.clientY;
  state.panStartX = state.panX;
  state.panStartY = state.panY;
  elements["scene-viewer"].setPointerCapture(event.pointerId);
  elements["scene-viewer"].classList.add("dragging");
});
elements["scene-viewer"].addEventListener("pointermove", (event) => {
  updateCursorCoordinate(event);
  if (!state.dragging || event.pointerId !== state.pointerId) return;
  const dx = event.clientX - state.dragStartX;
  const dy = event.clientY - state.dragStartY;
  state.dragMoved ||= Math.abs(dx) + Math.abs(dy) > 3;
  state.panX = state.panStartX + dx;
  state.panY = state.panStartY + dy;
  applyViewerTransform();
});
function endDrag(event) {
  if (!state.dragging || event.pointerId !== state.pointerId) return;
  state.dragging = false;
  state.pointerId = null;
  elements["scene-viewer"].classList.remove("dragging");
}
elements["scene-viewer"].addEventListener("pointerup", endDrag);
elements["scene-viewer"].addEventListener("pointercancel", endDrag);
elements["scene-viewer"].addEventListener("pointerleave", () => setText("cursor-coordinate", "—"));
elements["scene-viewer"].addEventListener("keydown", (event) => {
  if (event.key === "+" || event.key === "=") zoomViewer(1.25);
  if (event.key === "-") zoomViewer(0.8);
  if (event.key === "0") resetViewer();
});
window.addEventListener("resize", () => {
  window.clearTimeout(window.vitaResizeTimer);
  window.vitaResizeTimer = window.setTimeout(resetViewer, 100);
});

Promise.allSettled([checkService(), loadScenes({ preserveSelection: false })]);
