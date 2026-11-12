/* What Is That Smell? — simulated electronic nose front-end */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const COLORS = {
    ink: "#e9e4d8",
    muted: "#8b948c",
    faint: "#59635d",
    grid: "rgba(233, 228, 216, 0.09)",
    gridStrong: "rgba(233, 228, 216, 0.16)",
    amber: "#e3b04b",
    mint: "#7fe3c8",
    coral: "#ff9070",
  };

  const FAMILY_TONES = {
    sweet: "#e3b04b",
    fruity: "#f08b64",
    citrus: "#e7d55c",
    floral: "#e7a6c9",
    woody: "#b68a5a",
    green: "#8fce7c",
    spicy: "#e07a54",
    savory: "#c79b63",
    animal: "#a58c7d",
    chemical: "#8fb8c9",
    earthy: "#a89a6f",
    soapy: "#c9c8bc",
  };

  const state = {
    vials: [],
    sensors: [],
    selectedVial: null,
    customSmiles: null,
    result: null,
    animating: false,
    animationStart: 0,
    raf: 0,
    currentTime: null,
    lastDomUpdate: 0,
    maxPeak: 60,
    radarFamilies: null,
  };

  const els = {};

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    cacheElements();
    buildSkeletonSensors();
    bindEvents();
    observeCanvases();
    drawTrace(null);
    drawRadar(null);
    loadLibrary();
  }

  function cacheElements() {
    els.rack = $("#rack");
    els.rackCount = $("#rackCount");
    els.customToggle = $("#customToggle");
    els.customPanel = $("#customPanel");
    els.smilesInput = $("#smilesInput");
    els.loadSmiles = $("#loadSmiles");
    els.customError = $("#customError");
    els.ppm = $("#ppm");
    els.humidity = $("#humidity");
    els.temperature = $("#temperature");
    els.ppmLabel = $("#ppmLabel");
    els.humidityLabel = $("#humidityLabel");
    els.temperatureLabel = $("#temperatureLabel");
    els.sniffButton = $("#sniffButton");
    els.sniffSub = $("#sniffSub");
    els.sniffCopyStrong = $("#sniffButton strong");
    els.phaseLabel = $("#phaseLabel");
    els.chamberStage = $("#chamberStage");
    els.structureImage = $("#structureImage");
    els.moleculeEmpty = $("#moleculeEmpty");
    els.moleculeStamp = $("#moleculeStamp");
    els.identityName = $("#identityName");
    els.identityData = $("#identityData");
    els.referenceBadge = $("#referenceBadge");
    els.sensorArray = $("#sensorArray");
    els.traceChart = $("#traceChart");
    els.traceLegend = $("#traceLegend");
    els.radarChart = $("#radarChart");
    els.resultsPanel = $(".results-panel");
    els.resultStatus = $("#resultStatus");
    els.verdictWord = $("#verdictWord");
    els.verdictSub = $("#verdictSub");
    els.confidenceBar = $("#confidenceBar");
    els.confidenceCaption = $("#confidenceCaption");
    els.confidenceValue = $("#confidenceValue");
    els.noteList = $("#noteList");
    els.neighborList = $("#neighborList");
    els.bottleMatch = $("#bottleMatch");
    els.bottleName = $("#bottleName");
    els.bottleScore = $("#bottleScore");
    els.bottleBar = $("#bottleBar");
    els.readoutPeak = $("#readoutPeak");
    els.readoutState = $("#readoutState");
    els.readoutFlow = $("#readoutFlow");
    els.readoutHeater = $("#readoutHeater");
    els.aucValue = $("#aucValue");
    els.skillValue = $("#skillValue");
    els.onlineLabel = $("#onlineLabel");
    els.onlinePulse = $("#onlinePulse");
    els.footerModel = $("#footerModel");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function bindEvents() {
    els.ppm.addEventListener("input", () => {
      els.ppmLabel.textContent = `${els.ppm.value} ppm`;
      updateSniffSub();
    });
    els.humidity.addEventListener("input", () => {
      els.humidityLabel.textContent = `${els.humidity.value}%`;
    });
    els.temperature.addEventListener("input", () => {
      els.temperatureLabel.textContent = `${els.temperature.value}°C`;
    });

    els.customToggle.addEventListener("click", () => {
      const expanded = els.customToggle.getAttribute("aria-expanded") === "true";
      els.customToggle.setAttribute("aria-expanded", String(!expanded));
      els.customPanel.hidden = expanded;
      if (!expanded) els.smilesInput.focus();
    });

    els.loadSmiles.addEventListener("click", loadCustomMolecule);
    els.smilesInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadCustomMolecule();
      }
    });
    $$(".example").forEach((button) => {
      button.addEventListener("click", () => {
        els.smilesInput.value = button.dataset.smiles;
        loadCustomMolecule();
      });
    });

    els.sniffButton.addEventListener("click", startSniff);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !els.customPanel.hidden) {
        els.customPanel.hidden = true;
        els.customToggle.setAttribute("aria-expanded", "false");
      }
    });

    els.traceChart.addEventListener("mousemove", (event) => {
      if (!state.result || state.animating) return;
      const rect = els.traceChart.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / rect.width;
      const times = state.result.sensors.times;
      const t = Math.max(0, Math.min(times[times.length - 1], ratio * times[times.length - 1]));
      drawTrace(state.result, t, true);
    });
    els.traceChart.addEventListener("mouseleave", () => {
      if (state.result) drawTrace(state.result, state.currentTime);
    });
  }

  function updateSniffSub() {
    els.sniffSub.textContent = `draw 50 ml · ${els.ppm.value} ppm · ${els.humidity.value}% RH`;
  }

  async function loadLibrary() {
    try {
      const response = await fetch("/api/library");
      if (!response.ok) throw new Error(`Library request failed (${response.status})`);
      const data = await response.json();
      state.vials = data.vials || [];
      state.sensors = data.sensors || [];
      buildSensors(state.sensors);
      renderRack();
      const model = data.model || {};
      els.rackCount.textContent = `${state.vials.length} vials`;
      els.aucValue.textContent = Number.isFinite(model.test_macro_roc_auc)
        ? model.test_macro_roc_auc.toFixed(3)
        : "—";
      els.skillValue.textContent = model.notes ?? "138";
      els.footerModel.textContent = `${model.molecules ? model.molecules.toLocaleString() : "—"} molecules · ${model.notes || 138} notes`;
      els.onlinePulse.classList.remove("offline");
      els.onlineLabel.textContent = "ARRAY ONLINE";
      if (state.vials.length) selectVial(state.vials[0], true);
    } catch (error) {
      els.onlinePulse.classList.add("offline");
      els.onlineLabel.textContent = "ARRAY OFFLINE";
      els.rackCount.textContent = "no link";
      els.rack.innerHTML = `<p class="placeholder">Could not reach the nose.<br>Start the server with<br><code>python -m uvicorn what_is_that_smell.server:app</code>.</p>`;
      setPhase("OFFLINE", false);
      showError(error.message || "The nose is offline.");
    }
  }

  function buildSkeletonSensors() {
    const skeleton = Array.from({ length: 16 }, (_, index) => ({
      id: `EN-${String(index + 1).padStart(2, "0")}`,
      film: "MOS",
      target: "warming up",
      color: "#5b6b63",
    }));
    buildSensors(skeleton);
  }

  function buildSensors(sensors) {
    if (!els.sensorArray || !sensors.length) return;
    els.sensorArray.innerHTML = "";
    sensors.forEach((sensor) => {
      const cell = document.createElement("div");
      cell.className = "sensor";
      cell.style.setProperty("--c", sensor.color);
      cell.title = `${sensor.id} · ${sensor.film} · targets ${sensor.target}`;
      cell.innerHTML = `
        <div class="sensor-top">
          <span class="sensor-id">${escapeHtml(sensor.id)}</span>
          <span class="sensor-dot"></span>
        </div>
        <span class="sensor-value">0.0</span>
        <span class="sensor-film">${escapeHtml(sensor.film)}</span>`;
      els.sensorArray.appendChild(cell);
    });
  }

  function renderRack() {
    els.rack.innerHTML = "";
    state.vials.forEach((vial) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "vial";
      button.dataset.id = vial.id;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", "false");
      button.style.setProperty("--tone", FAMILY_TONES[vial.family] || "var(--amber)");
      button.innerHTML = `
        <span class="vial-name">${escapeHtml(vial.name)}</span>
        <span class="vial-compound">${escapeHtml(vial.compound)}</span>`;
      button.title = `${vial.name} — ${vial.compound} (${vial.formula})`;
      button.addEventListener("click", () => selectVial(vial));
      els.rack.appendChild(button);
    });
  }

  function selectVial(vial, silent = false) {
    if (state.animating) return;
    state.selectedVial = vial;
    state.customSmiles = null;
    $$(".vial", els.rack).forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.id === vial.id));
    });
    applySamplePreview(vial);
    resetReadout();
    setPhase("READY · VESSEL LOADED", false);
    if (!silent) els.customError.textContent = "";
    updateSniffSub();
  }

  function applySamplePreview(sample) {
    if (sample.structure_png) {
      els.structureImage.src = sample.structure_png;
      els.structureImage.hidden = false;
      els.moleculeEmpty.hidden = true;
    } else {
      els.structureImage.hidden = true;
      els.structureImage.removeAttribute("src");
      els.moleculeEmpty.hidden = false;
    }
    els.moleculeStamp.hidden = false;
    els.moleculeStamp.textContent = sample.in_reference ? "REF" : "NEW";
    els.moleculeStamp.style.color = sample.in_reference ? "var(--mint)" : "var(--amber)";

    els.identityName.textContent = sample.name || "Unknown compound";
    const bits = [sample.formula, sample.mw ? `${Number(sample.mw).toFixed(1)} g/mol` : null];
    if (sample.logp !== undefined) bits.push(`logP ${sample.logp}`);
    if (sample.compound) bits.push(sample.compound);
    els.identityData.textContent = bits.filter(Boolean).join(" · ");

    els.referenceBadge.hidden = false;
    if (sample.in_reference) {
      const split = sample.reference_split
        ? ` · ${String(sample.reference_split).toUpperCase()}`
        : "";
      els.referenceBadge.textContent = `IN REFERENCE SET${split}`;
      els.referenceBadge.style.color = "var(--mint)";
      els.referenceBadge.style.borderColor = "rgba(127, 227, 200, 0.35)";
    } else {
      els.referenceBadge.textContent = "NOVEL STRUCTURE";
      els.referenceBadge.style.color = "var(--amber)";
      els.referenceBadge.style.borderColor = "rgba(227, 176, 75, 0.4)";
    }
  }

  async function loadCustomMolecule() {
    if (state.animating) return;
    const smiles = els.smilesInput.value.trim();
    if (!smiles) {
      els.customError.textContent = "Enter a SMILES string first.";
      return;
    }
    els.customError.textContent = "";
    els.loadSmiles.disabled = true;
    try {
      const response = await fetch("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not parse that molecule.");
      state.selectedVial = null;
      state.customSmiles = data.smiles;
      $$(".vial", els.rack).forEach((button) => button.setAttribute("aria-selected", "false"));
      applySamplePreview(data);
      resetReadout();
      setPhase("READY · CUSTOM MOLECULE", false);
      els.smilesInput.value = data.smiles;
    } catch (error) {
      els.customError.textContent = error.message;
    } finally {
      els.loadSmiles.disabled = false;
    }
  }

  async function startSniff() {
    if (state.animating) return;
    if (!state.selectedVial && !state.customSmiles) {
      els.customError.textContent = "Choose a vial or load a custom molecule first.";
      return;
    }
    cancelAnimationFrame(state.raf);
    resetReadout();
    const requestBody = {
      ppm: Number(els.ppm.value),
      humidity: Number(els.humidity.value),
      temperature: Number(els.temperature.value),
    };
    if (state.selectedVial) requestBody.vial_id = state.selectedVial.id;
    else requestBody.smiles = state.customSmiles;

    setBusy(true);
    els.resultStatus.textContent = "SAMPLING";
    try {
      const response = await fetch("/api/sniff", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The nose failed to read this sample.");
      state.result = data;
      state.maxPeak = Math.max(
        20,
        ...data.sensors.channels.map((channel) => channel.peak)
      );
      applySamplePreview(data.sample);
      setPhase("PURGE · BASELINE", true);
      animateSniff(data);
    } catch (error) {
      setBusy(false);
      setPhase("IDLE · VALVE CLOSED", false);
      showError(error.message || "The sniff failed.");
    }
  }

  function animateSniff(data) {
    state.animating = true;
    state.animationStart = performance.now();
    state.lastDomUpdate = 0;
    state.currentTime = 0;
    els.chamberStage.classList.add("is-sampling");
    els.sniffCopyStrong.textContent = "Sampling…";
    els.sniffSub.textContent = "reading volatile signature";
    els.sniffButton.disabled = true;
    els.resultsPanel.classList.remove("revealed");
    const duration = data.sensors.times[data.sensors.times.length - 1];
    renderTraceLegend(data.sensors.channels);

    const frame = (now) => {
      const elapsed = (now - state.animationStart) / 1000;
      const t = Math.min(elapsed, duration);
      state.currentTime = t;
      updateSensorsAt(t);
      drawTrace(data, t);

      if (t < data.sensors.exposure_start_s) {
        setPhase("PURGE · BASELINE", true);
      } else if (t < data.sensors.exposure_end_s) {
        setPhase("SAMPLING · VALVE OPEN", true);
      } else if (t < duration) {
        setPhase("PURGE · DESORBING", true);
      }

      if (t < duration) {
        state.raf = requestAnimationFrame(frame);
      } else {
        finishSniff(data);
      }
    };
    state.raf = requestAnimationFrame(frame);
  }

  function finishSniff(data) {
    state.animating = false;
    state.currentTime = data.sensors.times[data.sensors.times.length - 1];
    els.chamberStage.classList.remove("is-sampling");
    setPhase("COMPLETE · SIGNATURE CAPTURED", false);
    setBusy(false);
    els.sniffCopyStrong.textContent = "Sniff again";
    updateSensorPeaks(data.sensors.channels);
    drawTrace(data, null);
    els.readoutPeak.textContent = `${Math.max(...data.sensors.channels.map((c) => c.peak)).toFixed(1)}%`;
    els.readoutState.textContent = "ANALYZED";
    renderAnalysis(data);
    els.resultsPanel.classList.add("revealed");
    els.resultStatus.textContent = "SIGNATURE DECODED";
  }

  function setBusy(busy) {
    els.sniffButton.disabled = busy;
    els.sniffButton.setAttribute("aria-busy", String(busy));
    if (!busy) {
      els.sniffCopyStrong.textContent = state.result ? "Sniff again" : "Start sniff";
      updateSniffSub();
    }
  }

  function renderTraceLegend(channels) {
    const top = [...channels].sort((a, b) => b.peak - a.peak).slice(0, 3);
    els.traceLegend.innerHTML = top
      .map(
        (channel) =>
          `<span class="legend-item"><span class="legend-swatch" style="background:${channel.color}"></span>${escapeHtml(channel.id)}</span>`
      )
      .join("");
  }

  function updateSensorsAt(t) {
    if (!state.result) return;
    const now = performance.now();
    if (now - state.lastDomUpdate < 34) return;
    state.lastDomUpdate = now;
    const { times, channels } = state.result.sensors;
    const index = Math.min(times.length - 1, Math.max(0, Math.round(t / (times[1] - times[0] || 0.08))));
    const cells = $$(".sensor", els.sensorArray);
    let livePeak = 0;
    channels.forEach((channel, i) => {
      const cell = cells[i];
      if (!cell) return;
      const value = channel.trace[index];
      livePeak = Math.max(livePeak, value);
      cell.style.setProperty("--level", Math.min(1, value / 55).toFixed(3));
      const valueEl = $(".sensor-value", cell);
      if (valueEl) valueEl.textContent = value.toFixed(1);
    });
    els.readoutPeak.textContent = `${livePeak.toFixed(1)}%`;
    els.readoutState.textContent =
      t < state.result.sensors.exposure_start_s
        ? "BASELINE"
        : t < state.result.sensors.exposure_end_s
          ? "SAMPLING"
          : "PURGING";
  }

  function updateSensorPeaks(channels) {
    const cells = $$(".sensor", els.sensorArray);
    channels.forEach((channel, i) => {
      const cell = cells[i];
      if (!cell) return;
      cell.style.setProperty("--level", Math.min(1, channel.peak / 55).toFixed(3));
      const valueEl = $(".sensor-value", cell);
      if (valueEl) valueEl.textContent = channel.peak.toFixed(1);
    });
  }

  function setPhase(text, sampling) {
    els.phaseLabel.textContent = text;
    els.phaseLabel.classList.toggle("sampling", Boolean(sampling));
  }

  function renderAnalysis(data) {
    const analysis = data.analysis;
    const top = analysis.top_notes.slice(0, 6);
    els.verdictWord.textContent = analysis.verdict;
    els.verdictSub.textContent = top.slice(0, 3).map((note) => note.odor).join(" · ");
    const percent = Math.round(analysis.confidence * 100);
    els.confidenceBar.style.width = `${Math.min(100, Math.max(2, percent))}%`;
    els.confidenceValue.textContent = `${percent}%`;
    els.confidenceCaption.textContent = "top-note model confidence";

    els.noteList.innerHTML = "";
    const maxProbability = Math.max(0.01, ...top.map((note) => note.probability));
    top.forEach((note, index) => {
      const row = document.createElement("div");
      row.className = `note${index === 0 ? " first" : ""}`;
      row.innerHTML = `
        <span class="note-name">${escapeHtml(note.odor)}</span>
        <span class="note-track"><span class="note-fill"></span></span>
        <span class="note-score">${(note.probability * 100).toFixed(0)}%</span>`;
      els.noteList.appendChild(row);
      const fill = $(".note-fill", row);
      requestAnimationFrame(() => {
        fill.style.width = `${Math.max(2, (note.probability / maxProbability) * 100)}%`;
      });
    });

    els.neighborList.innerHTML = "";
    (data.neighbors || []).slice(0, 4).forEach((neighbor) => {
      const row = document.createElement("div");
      row.className = "neighbor";
      const label = neighbor.name || "Reference molecule";
      row.innerHTML = `
        <span class="neighbor-score">${(neighbor.similarity * 100).toFixed(1)}</span>
        <span class="neighbor-body">
          <span class="neighbor-name">${escapeHtml(label)}</span>
          <span class="neighbor-smiles" title="${escapeHtml(neighbor.smiles)}">${escapeHtml(neighbor.smiles)}</span>
        </span>
        <span class="split-tag">${escapeHtml(neighbor.split)}</span>`;
      els.neighborList.appendChild(row);
    });
    if (!els.neighborList.children.length) {
      els.neighborList.innerHTML = '<p class="placeholder">No near neighbours in the reference set.</p>';
    }

    const bottle = (data.closest_vials || [])[0];
    if (bottle) {
      els.bottleMatch.hidden = false;
      els.bottleName.textContent = bottle.name;
      els.bottleScore.textContent = `${Math.round(bottle.similarity * 100)}% match`;
      els.bottleBar.style.width = "0%";
      requestAnimationFrame(() => {
        els.bottleBar.style.width = `${Math.max(3, bottle.similarity * 100)}%`;
      });
    } else {
      els.bottleMatch.hidden = true;
    }

    state.radarFamilies = analysis.families;
    drawRadar(analysis.families);
  }

  function resetReadout() {
    if (state.raf) cancelAnimationFrame(state.raf);
    state.animating = false;
    state.currentTime = null;
    state.result = null;
    state.radarFamilies = null;
    els.chamberStage.classList.remove("is-sampling");
    els.resultsPanel.classList.remove("revealed");
    els.resultStatus.textContent = "AWAITING SAMPLE";
    els.verdictWord.textContent = "—";
    els.verdictSub.textContent = "No odor signature yet.";
    els.confidenceBar.style.width = "0%";
    els.confidenceCaption.textContent = "select and sniff a sample";
    els.confidenceValue.textContent = "";
    els.noteList.innerHTML = '<p class="placeholder">Run a sniff to decode the molecule.</p>';
    els.neighborList.innerHTML = '<p class="placeholder">—</p>';
    els.bottleMatch.hidden = true;
    els.traceLegend.innerHTML = "";
    els.readoutPeak.textContent = "—";
    els.readoutState.textContent = "READY";
    els.sniffButton.disabled = false;
    els.sniffCopyStrong.textContent = "Start sniff";
    updateSniffSub();
    $$(".sensor", els.sensorArray).forEach((cell) => {
      cell.style.setProperty("--level", "0");
      const valueEl = $(".sensor-value", cell);
      if (valueEl) valueEl.textContent = "0.0";
    });
    drawTrace(null);
    drawRadar(null);
  }

  function showError(message) {
    els.resultStatus.textContent = "ERROR";
    els.noteList.innerHTML = `<p class="placeholder" style="color:var(--coral)">${escapeHtml(message)}</p>`;
  }

  // ---------- canvas helpers ----------
  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function observeCanvases() {
    const redraw = () => {
      drawTrace(state.result, state.animating ? state.currentTime : null);
      drawRadar(state.radarFamilies);
    };
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(redraw);
      observer.observe($(".trace-canvas-wrap"));
      observer.observe($(".radar-wrap"));
    } else {
      window.addEventListener("resize", redraw);
    }
  }

  function drawTrace(data, time, hover = false) {
    if (!els.traceChart) return;
    const { ctx, width, height } = fitCanvas(els.traceChart);
    ctx.clearRect(0, 0, width, height);
    const pad = { left: 32, right: 8, top: 12, bottom: 19 };
    const plotW = Math.max(10, width - pad.left - pad.right);
    const plotH = Math.max(10, height - pad.top - pad.bottom);
    const yMax = data ? Math.max(20, Math.ceil(state.maxPeak / 10) * 10) : 60;
    const duration = data ? data.sensors.times[data.sensors.times.length - 1] : 9.2;

    const xFor = (t) => pad.left + (t / duration) * plotW;
    const yFor = (v) => pad.top + plotH - (v / yMax) * plotH;

    ctx.font = '7px "SFMono-Regular", Consolas, monospace';
    ctx.fillStyle = COLORS.faint;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i += 1) {
      const value = (yMax / 4) * i;
      const y = yFor(value);
      ctx.strokeStyle = i === 0 ? COLORS.gridStrong : COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y + 0.5);
      ctx.lineTo(pad.left + plotW, y + 0.5);
      ctx.stroke();
      ctx.fillText(String(Math.round(value)), pad.left - 6, y);
    }

    if (data) {
      const startX = xFor(data.sensors.exposure_start_s);
      const endX = xFor(data.sensors.exposure_end_s);
      ctx.fillStyle = "rgba(227, 176, 75, 0.045)";
      ctx.fillRect(startX, pad.top, endX - startX, plotH);
      ctx.strokeStyle = "rgba(227, 176, 75, 0.22)";
      ctx.setLineDash([2, 3]);
      [startX, endX].forEach((x) => {
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + plotH);
        ctx.stroke();
      });
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(227, 176, 75, 0.55)";
      ctx.textAlign = "left";
      ctx.fillText("SAMPLE", startX + 3, pad.top + 6);
      ctx.fillText("PURGE", endX + 3, pad.top + 6);
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = COLORS.faint;
    for (let seconds = 0; seconds <= Math.floor(duration); seconds += 2) {
      ctx.fillText(`${seconds}s`, xFor(seconds), pad.top + plotH + 5);
    }

    if (!data) {
      ctx.fillStyle = COLORS.faint;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = '7.5px "SFMono-Regular", Consolas, monospace';
      ctx.fillText("AWAITING SAMPLE", pad.left + plotW / 2, pad.top + plotH / 2);
      return;
    }

    const { times, channels } = data.sensors;
    const cutoff = time === null ? times.length - 1 : Math.max(0, Math.min(times.length - 1, Math.round(time / (times[1] - times[0] || 0.08))));
    const topIds = new Set([...channels].sort((a, b) => b.peak - a.peak).slice(0, 3).map((c) => c.id));

    channels.forEach((channel) => {
      const isTop = topIds.has(channel.id);
      ctx.beginPath();
      for (let i = 0; i <= cutoff; i += 1) {
        const x = xFor(times[i]);
        const y = yFor(Math.min(yMax, channel.trace[i]));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = channel.color;
      ctx.globalAlpha = isTop ? 0.92 : 0.22;
      ctx.lineWidth = isTop ? 1.5 : 0.9;
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    if (time !== null && !hover) {
      const cursorX = xFor(Math.min(time, duration));
      ctx.strokeStyle = "rgba(233, 228, 216, 0.32)";
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(cursorX, pad.top);
      ctx.lineTo(cursorX, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (hover) {
      const cursorX = xFor(Math.min(time, duration));
      ctx.strokeStyle = "rgba(237, 230, 214, 0.55)";
      ctx.beginPath();
      ctx.moveTo(cursorX, pad.top);
      ctx.lineTo(cursorX, pad.top + plotH);
      ctx.stroke();
      const index = Math.max(0, Math.min(times.length - 1, Math.round(time / (times[1] - times[0] || 0.08))));
      let best = null;
      channels.forEach((channel) => {
        if (!best || channel.trace[index] > best.trace[index]) best = channel;
      });
      if (best) {
        ctx.fillStyle = COLORS.amber;
        ctx.beginPath();
        ctx.arc(cursorX, yFor(best.trace[index]), 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = '7px "SFMono-Regular", Consolas, monospace';
        ctx.textAlign = cursorX > width * 0.7 ? "right" : "left";
        ctx.textBaseline = "bottom";
        ctx.fillStyle = COLORS.muted;
        const label = `${best.id} ${best.trace[index].toFixed(1)}% @ ${time.toFixed(1)}s`;
        ctx.fillText(label, cursorX + (cursorX > width * 0.7 ? -5 : 5), pad.top + plotH - 4);
      }
    }
  }

  function drawRadar(families) {
    if (!els.radarChart) return;
    const { ctx, width, height } = fitCanvas(els.radarChart);
    ctx.clearRect(0, 0, width, height);
    const fallback = ["citrus", "fruity", "floral", "sweet", "woody", "savory"];
    const source = Array.isArray(families) && families.length
      ? families.slice(0, 6)
      : fallback.map((family) => ({ family, score: 0 }));
    const labels = source.map((item) => item.family);
    const values = source.map((item) => Math.min(1, Number(item.score) || 0));
    const cx = width / 2;
    const cy = height / 2 + 3;
    const radius = Math.max(28, Math.min(width, height) * 0.31);
    const count = labels.length;
    const angle = (i) => -Math.PI / 2 + (i / count) * Math.PI * 2;
    const point = (i, r) => [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];

    ctx.font = '7px "SFMono-Regular", Consolas, monospace';
    for (let ring = 1; ring <= 3; ring += 1) {
      const r = (radius * ring) / 3;
      ctx.beginPath();
      for (let i = 0; i < count; i += 1) {
        const [x, y] = point(i, r);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = ring === 3 ? COLORS.gridStrong : COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (let i = 0; i < count; i += 1) {
      const [x, y] = point(i, radius);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = COLORS.grid;
      ctx.stroke();
    }

    ctx.beginPath();
    for (let i = 0; i < count; i += 1) {
      const [x, y] = point(i, Math.max(0.015, values[i]) * radius);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(227, 176, 75, 0.14)";
    ctx.fill();
    ctx.strokeStyle = COLORS.amber;
    ctx.lineWidth = 1.3;
    ctx.stroke();

    for (let i = 0; i < count; i += 1) {
      const [x, y] = point(i, Math.max(0.015, values[i]) * radius);
      ctx.beginPath();
      ctx.arc(x, y, 1.7, 0, Math.PI * 2);
      ctx.fillStyle = values[i] > 0.05 ? COLORS.amber : COLORS.faint;
      ctx.fill();

      const [lx, ly] = point(i, radius + 15);
      ctx.fillStyle = values[i] > 0.45 ? COLORS.ink : COLORS.faint;
      ctx.textAlign = Math.abs(lx - cx) < 4 ? "center" : lx > cx ? "left" : "right";
      ctx.textBaseline = "middle";
      ctx.fillText(labels[i], lx, ly);
    }
  }
})();
