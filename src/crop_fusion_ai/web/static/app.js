const state = {
  config: null,
  latestResult: null,
  activeTab: "overview",
  activeGroup: null,
  chat: {
    open: false,
    models: [],
    selectedModel: "",
    modelInfo: null,
    messages: [],
    loading: false,
    error: "",
  },
};

const el = (selector) => document.querySelector(selector);

function formatValue(value, display = "decimal") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Not available";
  }
  const numeric = Number(value);
  if (display === "percent") return `${(numeric * 100).toFixed(1)}%`;
  if (display === "integer") return Math.round(numeric).toLocaleString();
  return numeric.toFixed(2);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Not available";
  }
  return `${Number(value).toFixed(1)}%`;
}

function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.trim().split(/\s+/).filter(Boolean).length * 1.3));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function buildChatContextSnapshot(result) {
  if (!result) return {};
  return {
    headline: result.headline || {},
    summary: {
      best_model: result.summary?.best_model || {},
      holdout: result.summary?.holdout || {},
    },
    drivers: (result.drivers || []).slice(0, 3),
    feature_groups: (result.feature_groups || []).map((group) => ({
      group: group.group,
      label: group.label,
      features: (group.features || []).slice(0, 2).map((feature) => ({
        label: feature.label,
        description: feature.description,
        latest_value: feature.latest_value,
        recent: (feature.series || []).slice(-3).map((item) => item.value),
      })),
    })),
    yield_series: (result.yield_series || []).slice(-6).map((item) => ({
      month_label: item.month_label,
      predicted_yield: item.predicted_yield,
    })),
    monthly_features: (result.monthly_features || []).slice(-4).map((item) => ({
      month_label: item.month_label,
      predicted_yield: item.predicted_yield,
    })),
    feature_importance: (result.feature_importance || []).slice(0, 5).map((item) => ({
      label: item.label,
      importance: item.importance,
    })),
  };
}

function renderChatContext() {
  const target = el("#chatContextSummary");
  if (!target) return;
  const headlineMonth = state.latestResult?.headline?.month_label
    || (state.latestResult?.headline?.year && state.latestResult?.headline?.month
      ? `${state.latestResult.headline.year}-${String(state.latestResult.headline.month).padStart(2, "0")}`
      : null);
  const headline = headlineMonth || "No prediction run yet";
  const topDriver = state.latestResult?.drivers?.[0]?.label || "Not available";
  const rows = [
    `Latest run: ${headline}`,
    `Top driver: ${topDriver}`,
    `Monthly rows: ${(state.latestResult?.monthly_features || []).length}`,
  ];
  target.innerHTML = rows.map((item) => `<div>${item}</div>`).join("");
}

function renderChatMessages() {
  const target = el("#chatMessages");
  if (!target) return;
  const messages = state.chat.messages.length
    ? state.chat.messages
    : [
        {
          role: "system",
          content: "Ask about the current yield estimate, crop drivers, weather patterns, or what the model is seeing.",
        },
      ];
  target.innerHTML = messages
    .map(
      (message) => `
        <div class="chat-message ${message.role}">
          <div class="chat-role">${escapeHtml(message.role === "assistant" ? "TaoCrop" : message.role === "system" ? "System" : "You")}</div>
          <div class="chat-content">${escapeHtml(message.content)}</div>
        </div>
      `,
    )
    .join("");
  target.scrollTop = target.scrollHeight;
}

function updateChatControls() {
  const input = el("#chatInput");
  const select = el("#chatModelSelect");
  if (input) input.disabled = state.chat.loading;
  if (select) select.disabled = state.chat.loading;
}

function renderChatStatusLine() {
  const target = el("#chatStatusLine");
  if (!target) return;
  const info = state.chat.modelInfo || {};
  const contextLength = info.context_length || "Not reported";
  const lastStats = state.chat.lastStats || {};
  const currentTokens = lastStats.input_tokens ?? estimateTokens(JSON.stringify(buildChatContextSnapshot(state.latestResult)));
  const tokensPerSecond = lastStats.tokens_per_second;
  target.innerHTML = `
    <strong>${state.chat.selectedModel || "Select a model"}</strong>
    <span>Context ${contextLength}</span>
    <span>Current ${currentTokens} tokens</span>
    <span>${tokensPerSecond && Number.isFinite(tokensPerSecond) ? `${tokensPerSecond.toFixed(1)} tok/s` : "tok/s after reply"}</span>
    <span>Enter sends</span>
    <span>Ctrl+Enter newline</span>
  `;
}

function setChatOpen(open) {
  state.chat.open = open;
  document.body.classList.toggle("chat-open", open);
  el("#chatBackdrop").classList.toggle("hidden", !open);
  el("#chatPanel").classList.toggle("hidden", !open);
  if (open) {
    renderChatContext();
    renderChatMessages();
    renderChatStatusLine();
    setTimeout(() => el("#chatInput").focus(), 0);
  }
}

async function loadChatModels() {
  const response = await fetch("/api/chat/models");
  if (!response.ok) {
    throw new Error("Could not load Ollama models");
  }
  const payload = await response.json();
  state.chat.models = payload.models || [];
  const select = el("#chatModelSelect");
  if (!state.chat.models.length) {
    select.innerHTML = `<option value="">No Ollama models found</option>`;
    state.chat.selectedModel = "";
    state.chat.modelInfo = null;
    updateChatControls();
    renderChatStatusLine();
    return;
  }
  select.innerHTML = state.chat.models
    .map((model) => {
      const name = model.model || model.name;
      return `<option value="${name}">${name}</option>`;
    })
    .join("");
  if (!state.chat.selectedModel) {
    state.chat.selectedModel = state.chat.models[0].model || state.chat.models[0].name;
  }
  select.value = state.chat.selectedModel;
  if (state.chat.selectedModel) {
    await syncChatModelInfo(state.chat.selectedModel);
  }
}

async function syncChatModelInfo(model) {
  const response = await fetch(`/api/chat/models/${encodeURIComponent(model)}`);
  if (!response.ok) {
    throw new Error("Could not load model details");
  }
  state.chat.modelInfo = await response.json();
  updateChatControls();
  renderChatStatusLine();
}

function buildChatMessages() {
  return state.chat.messages.map((message) => ({ role: message.role, content: message.content }));
}

async function submitChat(event) {
  event.preventDefault();
  const input = el("#chatInput");
  const text = input.value.trim();
  if (!text || state.chat.loading) return;
  if (!state.chat.selectedModel) {
    throw new Error("Select an Ollama model first.");
  }

  state.chat.messages.push({ role: "user", content: text });
  state.chat.messages = state.chat.messages.slice(-8);
  input.value = "";
  state.chat.loading = true;
  state.chat.error = "";
  updateChatControls();
  renderChatMessages();
  renderChatStatusLine();

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: state.chat.selectedModel,
      messages: buildChatMessages(),
      dashboard_context: buildChatContextSnapshot(state.latestResult),
    }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Chat failed");
  }
  const payload = await response.json();
  state.chat.messages.push({ role: "assistant", content: payload.reply || "No response returned." });
  state.chat.messages = state.chat.messages.slice(-8);
  state.chat.lastStats = payload.stats || {};
  state.chat.loading = false;
  updateChatControls();
  state.chat.modelInfo = {
    ...(state.chat.modelInfo || {}),
    context_length: payload.stats?.context_length || state.chat.modelInfo?.context_length,
  };
  renderChatMessages();
  renderChatStatusLine();
}

function handleChatComposerKeydown(event) {
  if (event.key !== "Enter") return;
  const input = event.currentTarget;
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    input.setRangeText("\n", start, end, "end");
    renderChatStatusLine();
    return;
  }
  event.preventDefault();
  const form = el("#chatForm");
  form.requestSubmit();
}

function resolveMonthLabel(row) {
  const label = row?.month_label;
  if (typeof label === "string" && label && label !== "Unknown" && !label.startsWith("1970-")) {
    return label;
  }
  if (row?.year === null || row?.year === undefined || row?.month === null || row?.month === undefined) {
    return "";
  }
  return `${row.year}-${String(row.month).padStart(2, "0")}`;
}

function setStatus(text, message) {
  el("#runStatus").textContent = text;
  el("#runMessage").textContent = message;
}

function showPanels(visible) {
  el("#resultGrid").classList.toggle("hidden", !visible);
  el("#tabs").classList.toggle("hidden", !visible);
  el("#panels").classList.toggle("hidden", !visible);
}

function buildHeroStats(config) {
  const stats = [
    {
      value: config.model?.best_model?.model ?? config.model?.model_name ?? "Model",
      label: "Best saved model",
    },
    {
      value: config.model?.holdout?.rmse ? formatValue(config.model.holdout.rmse) : "Ready",
      label: "Holdout RMSE",
    },
    {
      value: config.defaults?.crop_type ?? "corn",
      label: "Default crop",
    },
  ];

  el("#heroStats").innerHTML = stats
    .map(
      (item) => `
        <div class="hero-stat">
          <span class="value">${item.value}</span>
          <span class="label">${item.label}</span>
        </div>
      `,
    )
    .join("");
}

function renderSummaryList(result) {
  const rows = [
    {
      title: "Latest month analyzed",
      text: result.headline?.month_label ?? "Not available",
    },
    {
      title: "Monthly rows analyzed",
      text: `${result.monthly_features?.length ?? 0} monthly rows`,
    },
    {
      title: "Top driver",
      text: result.drivers?.[0]?.label ?? "Not available",
    },
    {
      title: "What it means",
      text: "The estimate combines crop, weather, and season signals to describe likely field performance.",
    },
  ];

  el("#summaryList").innerHTML = rows
    .map(
      (item) => `
        <div class="summary-item">
          <strong>${item.title}</strong>
          <div>${item.text}</div>
        </div>
      `,
    )
    .join("");
}

function svgSize(svg) {
  const viewBox = svg.getAttribute("viewBox");
  if (!viewBox) return { width: 800, height: 320 };
  const [, , width, height] = viewBox.split(/\s+/).map(Number);
  return { width, height };
}

function clearSvg(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function drawAxes(svg, width, height, padding) {
  svg.appendChild(svgEl("line", {
    x1: padding.left,
    y1: height - padding.bottom,
    x2: width - padding.right,
    y2: height - padding.bottom,
    stroke: "rgba(43, 33, 24, 0.15)",
    "stroke-width": 2,
  }));
  svg.appendChild(svgEl("line", {
    x1: padding.left,
    y1: padding.top,
    x2: padding.left,
    y2: height - padding.bottom,
    stroke: "rgba(43, 33, 24, 0.15)",
    "stroke-width": 2,
  }));
}

function drawLineChart(svg, series, options = {}) {
  clearSvg(svg);
  const { width, height } = svgSize(svg);
  const padding = { top: 18, right: 24, bottom: 42, left: 56, ...(options.padding || {}) };
  const showAxes = options.showAxes !== false;
  const showTicks = options.showTicks !== false;
  const showPoints = options.showPoints !== false;
  const showXAxisLabels = options.showXAxisLabels !== false;
  const valueScale = options.valueScale ?? 1;
  const tickFormatter = options.tickFormatter || ((value) => value.toFixed(1));
  const splitIndex = options.splitIndex ?? null;
  const leftStroke = options.leftStroke || "#d9772b";
  const rightStroke = options.rightStroke || "#7c4fd4";
  const leftPointFill = options.leftPointFill || "#fff7ef";
  const rightPointFill = options.rightPointFill || "#f2eaff";
  if (showXAxisLabels) {
    padding.bottom = Math.max(padding.bottom, 50);
  }
  if (showAxes) drawAxes(svg, width, height, padding);
  if (!series.length) {
    svg.appendChild(svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#756457" })).textContent = "No data";
    return;
  }

  const rawValues = series.map((item) => {
    const numeric = Number(item.value);
    return Number.isFinite(numeric) ? numeric * valueScale : null;
  });
  const values = rawValues.filter((value) => value !== null);
  if (!values.length) {
    svg.appendChild(svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#756457" })).textContent = "No data";
    return;
  }

  const filledValues = rawValues.slice();
  let lastSeen = null;
  for (let index = 0; index < filledValues.length; index += 1) {
    if (filledValues[index] !== null) {
      lastSeen = filledValues[index];
      break;
    }
  }
  for (let index = 0; index < filledValues.length; index += 1) {
    if (filledValues[index] === null) {
      filledValues[index] = lastSeen;
    } else {
      lastSeen = filledValues[index];
    }
  }
  lastSeen = null;
  for (let index = filledValues.length - 1; index >= 0; index -= 1) {
    if (filledValues[index] === null) {
      filledValues[index] = lastSeen;
    } else {
      lastSeen = filledValues[index];
    }
  }
  const fallbackValue = values[0];
  const normalizedValues = filledValues.map((value) => (value === null ? fallbackValue : value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const step = innerWidth / Math.max(series.length - 1, 1);

  const points = series.map((item, index) => {
    const x = padding.left + step * index;
    const value = normalizedValues[index];
    const y = padding.top + innerHeight - ((value - min) / span) * innerHeight;
    return { x, y, label: item.label, value };
  });

  const areaPath = [
    `M ${points[0].x} ${height - padding.bottom}`,
    `L ${points[0].x} ${points[0].y}`,
    ...points.slice(1).map((point) => `L ${point.x} ${point.y}`),
    `L ${points[points.length - 1].x} ${height - padding.bottom}`,
    "Z",
  ].join(" ");

  const defs = svgEl("defs");
  defs.appendChild(svgEl("linearGradient", { id: `${options.id || "gradient"}-fill`, x1: "0%", y1: "0%", x2: "0%", y2: "100%" }));
  defs.firstChild.appendChild(svgEl("stop", { offset: "0%", "stop-color": "rgba(217,119,43,0.45)" }));
  defs.firstChild.appendChild(svgEl("stop", { offset: "100%", "stop-color": "rgba(217,119,43,0.02)" }));
  defs.appendChild(svgEl("linearGradient", { id: `${options.id || "gradient"}-line`, x1: "0%", y1: "0%", x2: "100%", y2: "0%" }));
  defs.lastChild.appendChild(svgEl("stop", { offset: "0%", "stop-color": "#d9772b" }));
  defs.lastChild.appendChild(svgEl("stop", { offset: "100%", "stop-color": "#b94f0a" }));
  svg.appendChild(defs);
  const linePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  if (splitIndex !== null && splitIndex > 0 && splitIndex < points.length - 1) {
    const leftPath = points.slice(0, splitIndex + 1).map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    const rightPath = points.slice(splitIndex).map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    svg.appendChild(svgEl("path", { d: areaPath, fill: `url(#${options.id || "gradient"}-fill)`, opacity: 0.5 }));
    svg.appendChild(svgEl("path", {
      d: leftPath,
      fill: "none",
      stroke: leftStroke,
      "stroke-width": 4,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
    svg.appendChild(svgEl("path", {
      d: rightPath,
      fill: "none",
      stroke: rightStroke,
      "stroke-width": 4,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
  } else {
    svg.appendChild(svgEl("path", {
      d: areaPath,
      fill: `url(#${options.id || "gradient"}-fill)`,
    }));
    svg.appendChild(svgEl("path", {
      d: linePath,
      fill: "none",
      stroke: `url(#${options.id || "gradient"}-line)`,
      "stroke-width": 4,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
  }

  if (showPoints) {
    points.forEach((point) => {
      const pointColor = splitIndex !== null && splitIndex > 0 && splitIndex < points.length - 1 && point.x >= points[splitIndex].x
        ? rightStroke
        : leftStroke;
      const pointFill = splitIndex !== null && splitIndex > 0 && splitIndex < points.length - 1 && point.x >= points[splitIndex].x
        ? rightPointFill
        : leftPointFill;
      svg.appendChild(svgEl("circle", {
        cx: point.x,
        cy: point.y,
        r: 4.5,
        fill: pointFill,
        stroke: pointColor,
        "stroke-width": 2,
      }));
    });
  }

  if (showTicks) {
    const ticks = Math.min(4, series.length);
    for (let i = 0; i < ticks; i += 1) {
      const t = ticks === 1 ? 0 : i / (ticks - 1);
      const value = min + t * span;
      const y = padding.top + innerHeight - t * innerHeight;
      svg.appendChild(svgEl("text", {
        x: padding.left - 10,
        y: y + 4,
        "text-anchor": "end",
        fill: "#756457",
        "font-size": 12,
      })).textContent = tickFormatter(value);
    }
  }

  if (showAxes && showXAxisLabels) {
    points.forEach((point, index) => {
      if (point.label) {
        svg.appendChild(svgEl("text", {
          x: point.x,
          y: height - 4,
          "text-anchor": "middle",
          fill: "#756457",
          "font-size": 11,
        })).textContent = point.label;
      }
    });
  }
}

function drawBarChart(svg, items, options = {}) {
  clearSvg(svg);
  const { width, height } = svgSize(svg);
  const padding = { top: 18, right: 24, bottom: 30, left: 230 };
  drawAxes(svg, width, height, padding);
  if (!items.length) {
    svg.appendChild(svgEl("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#756457" })).textContent = "No data";
    return;
  }

  const max = Math.max(...items.map((item) => Number(item.value)));
  const innerWidth = width - padding.left - padding.right;
  const rowHeight = (height - padding.top - padding.bottom) / items.length;

  items.slice().reverse().forEach((item, index) => {
    const y = padding.top + index * rowHeight + rowHeight * 0.2;
    const barWidth = Math.max(1, (Number(item.value) / max) * innerWidth);
    svg.appendChild(svgEl("text", {
      x: padding.left - 14,
      y: y + rowHeight * 0.5,
      "text-anchor": "end",
      fill: "#2b2118",
      "font-size": 12,
    })).textContent = item.label;
    svg.appendChild(svgEl("rect", {
      x: padding.left,
      y,
      width: barWidth,
      height: rowHeight * 0.6,
      rx: 12,
      fill: `rgba(217, 119, 43, ${0.2 + (index / items.length) * 0.6})`,
    }));
    svg.appendChild(svgEl("text", {
      x: padding.left + barWidth + 10,
      y: y + rowHeight * 0.5,
      fill: "#2b2118",
      "font-size": 12,
    })).textContent = formatValue(item.value, options.display || "decimal");
  });
}

function renderFeatureGroups(result) {
  const groups = result.feature_groups || [];
  const tabs = [
    { group: "primary-driver", label: "Primary Driver" },
    ...groups.map((group) => ({ group: group.group, label: group.label })),
  ];
  if (!state.activeGroup) state.activeGroup = "primary-driver";

  el("#featureGroupTabs").innerHTML = tabs
    .map(
      (group) => `
        <button class="inner-tab ${group.group === state.activeGroup ? "active" : ""}" data-group="${group.group}">
          ${group.label}
        </button>
      `,
    )
    .join("");

  const allFeatures = groups.flatMap((group) => group.features || []);
  const drivers = (result.drivers || []).slice(0, 5);
  const driverList = drivers.length ? drivers.map((driver) => `<li>${driver.label}</li>`).join("") : "<li>Not available</li>";
  el("#primaryDriverSummary").innerHTML = `
    <div class="driver-summary-item">
      <strong>Primary driver</strong>
      <ul class="driver-summary-list">${driverList}</ul>
    </div>
  `;

  const primaryPanel = el("#primaryDriverPanel");
  primaryPanel.classList.toggle("hidden", state.activeGroup !== "primary-driver");
  const featurePanels = el("#featureGroupPanels");

  const importance = result.feature_importance || [];
  if (state.activeGroup === "primary-driver") {
    featurePanels.classList.add("hidden");
    featurePanels.innerHTML = "";
    drawBarChart(
      el("#importanceChart"),
      importance.slice(0, 8).map((item) => ({
        label: item.label,
        value: item.importance,
      })),
    );
    return;
  }

  if (!groups.length) {
    featurePanels.classList.add("hidden");
    featurePanels.innerHTML = "";
    return;
  }

  featurePanels.classList.remove("hidden");
  featurePanels.innerHTML = groups
    .map((group) => {
      const active = group.group === state.activeGroup ? "" : "hidden";
      const cards = group.features
        .map(
          (feature, index) => `
            <article class="feature-card">
              <div class="feature-card-top">
                <div>
                  <div class="feature-label">${feature.label}${feature.latest_value ? ` (Currently ${feature.latest_value})` : ""}</div>
                  <div class="feature-desc">${feature.description}</div>
                </div>
              </div>
              <svg class="spark" id="spark-${group.group}-${index}" viewBox="0 0 1200 220" preserveAspectRatio="xMidYMid meet"></svg>
            </article>
          `,
        )
        .join("");
      return `
        <section class="feature-group-card ${active}" data-group-panel="${group.group}">
          <div class="feature-grid">${cards}</div>
        </section>
      `;
    })
    .join("");

  groups.forEach((group) => {
    group.features.forEach((feature, index) => {
      const svg = document.getElementById(`spark-${group.group}-${index}`);
      if (!svg) return;
      const series = Array.isArray(feature.series) && feature.series.some((item) => item && item.value !== null && item.value !== undefined)
        ? feature.series
        : (result.monthly_features || []).map((row) => ({
            month_label: resolveMonthLabel(row),
            value: row?.[feature.name] ?? null,
          }));
      const cleanedSeries = series.map((item) => ({
        label: resolveMonthLabel(item),
        value: item.value,
      }));
      drawLineChart(
        svg,
        cleanedSeries,
        {
          id: `spark-${group.group}-${index}`,
          showAxes: true,
          showTicks: true,
          showPoints: true,
          showXAxisLabels: true,
          padding: { top: 16, right: 20, bottom: 24, left: 52 },
          valueScale: feature.display === "percent" ? 100 : 1,
          tickFormatter: feature.display === "percent" ? (value) => `${value.toFixed(0)}%` : (value) => value.toFixed(1),
          splitIndex: Math.max(1, Math.floor(cleanedSeries.length / 2)),
          leftStroke: "#d9772b",
          rightStroke: "#7c4fd4",
          leftPointFill: "#fff7ef",
          rightPointFill: "#f2eaff",
        },
      );
    });
  });
}

function renderBenchmarkTable(result) {
  const summary = result.summary || {};
  const best = summary.best_model || {};
  const holdout = summary.holdout || {};
  const rows = [
    ["Best model", best.model || summary.model_name || "Not available"],
    ["Holdout RMSE", holdout.rmse !== null && holdout.rmse !== undefined ? formatValue(holdout.rmse) : "Not available"],
    ["Holdout MAE", holdout.mae !== null && holdout.mae !== undefined ? formatValue(holdout.mae) : "Not available"],
    ["R squared", holdout.r2 !== null && holdout.r2 !== undefined ? formatValue(holdout.r2) : "Not available"],
  ];
  el("#benchmarkTable").innerHTML = `
    <table>
      <tbody>
        ${rows.map(([key, value]) => `<tr><th>${key}</th><td>${value}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderFeatureTable(result) {
  const rows = (result.monthly_features || []).slice(-8).reverse();
  const keys = rows.length
    ? ["month_label", "predicted_yield", "ag_green_pixel_ratio", "ndvi_mean", "weather_total_precipitation", "weather_gdd"]
    : [];
  const headers = {
    month_label: "Month",
    predicted_yield: "Predicted yield",
    ag_green_pixel_ratio: "Green canopy",
    ndvi_mean: "Vegetation vigor",
    weather_total_precipitation: "Rainfall",
    weather_gdd: "Growing degree days",
  };
  el("#featureTable").innerHTML = `
    <table>
      <thead>
        <tr>${keys.map((key) => `<th>${headers[key] || key}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                ${keys
                  .map((key) => {
                    const value = row[key];
                    if (key === "month_label") return `<td>${value ?? ""}</td>`;
                    if (typeof value === "number") return `<td>${value.toFixed(2)}</td>`;
                    return `<td>${value ?? ""}</td>`;
                  })
                  .join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderCharts(result) {
  const predictions = result.yield_series || [];
  const cleaned = predictions.map((item) => ({
    label: resolveMonthLabel(item),
    value: item.predicted_yield,
  }));
  drawLineChart(
    el("#yieldChart"),
    cleaned,
    {
      id: "yieldChart",
      splitIndex: Math.max(1, Math.floor(cleaned.length / 2)),
      leftStroke: "#d9772b",
      rightStroke: "#7c4fd4",
      showXAxisLabels: true,
      showPoints: true,
    },
  );

  drawLineChart(
    el("#trajectoryChart"),
    cleaned,
    {
      id: "trajectoryChart",
      splitIndex: Math.max(1, Math.floor(cleaned.length / 2)),
      leftStroke: "#d9772b",
      rightStroke: "#7c4fd4",
      showXAxisLabels: true,
      showPoints: true,
    },
  );
}

function renderHeadline(result) {
  const headline = result.headline || {};
  const latest = headline.month ? `${headline.year}-${String(headline.month).padStart(2, "0")}` : "Latest available month";
  el("#headlineYield").textContent = `${formatValue(headline.predicted_yield)} ${headline.unit || ""}`.trim();
  el("#headlineMonth").textContent = latest;
  const summary = result.summary || {};
  const best = summary.best_model || {};
  el("#modelQuality").textContent = best.rmse ? `RMSE ${formatValue(best.rmse)} | MAE ${formatValue(best.mae)}` : "Ready";
  el("#topDriver").textContent = result.drivers?.[0]?.label || "Not available";
  el("#dataCoverage").textContent = `${(result.monthly_features || []).length} monthly rows`;
}

function renderResult(result) {
  state.latestResult = result;
  showPanels(true);
  renderHeadline(result);
  renderSummaryList(result);
  renderCharts(result);
  renderBenchmarkTable(result);
  renderFeatureTable(result);
  renderFeatureGroups(result);
  renderChatContext();
  renderChatStatusLine();
  setStatus("Prediction complete", "The yield estimate, drivers, and grouped charts are ready.");
}

async function waitForJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      throw new Error(`Job ${jobId} not found`);
    }
    const job = await response.json();
    setStatus(job.status === "running" ? "Running" : job.status, job.message || "Working...");
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "Prediction failed");
    await new Promise((resolve) => setTimeout(resolve, 650));
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  const countyId = formData.get("county_id");
  const cropType = formData.get("crop_type");
  const files = el("#folderInput").files;
  if (!files || files.length === 0) {
    throw new Error("Choose a folder of farm files first.");
  }

  const upload = new FormData();
  upload.set("county_id", countyId);
  upload.set("crop_type", cropType);
  const paths = [];
  Array.from(files).forEach((file) => {
    upload.append("files", file);
    paths.push(file.webkitRelativePath || file.name);
  });
  upload.set("relative_paths", JSON.stringify(paths));

  setStatus("Uploading", "Sending files to the Python backend...");
  const response = await fetch("/api/predict/upload", {
    method: "POST",
    body: upload,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Upload failed");
  }
  const job = await response.json();
  const result = await waitForJob(job.job_id);
  renderResult(result);
}

function wireTabs() {
  el("#tabs").addEventListener("click", (event) => {
    const button = event.target.closest(".tab-button");
    if (!button) return;
    state.activeTab = button.dataset.tab;
    document.querySelectorAll(".tab-button").forEach((node) => {
      node.classList.toggle("active", node.dataset.tab === state.activeTab);
    });
    document.querySelectorAll(".panel").forEach((node) => {
      const active = node.id.startsWith(`${state.activeTab}Panel`);
      node.classList.toggle("active", active);
      node.classList.toggle("hidden", !active);
    });
  });

  el("#featureGroupTabs").addEventListener("click", (event) => {
    const button = event.target.closest(".inner-tab");
    if (!button) return;
    state.activeGroup = button.dataset.group;
    renderFeatureGroups(state.latestResult);
  });
}

function wireChat() {
  el("#chatOpenButton").addEventListener("click", () => setChatOpen(true));
  el("#chatBackdrop").addEventListener("click", () => setChatOpen(false));
  el("#chatCloseButton").addEventListener("click", () => setChatOpen(false));
  el("#chatModelSelect").addEventListener("change", (event) => {
    state.chat.selectedModel = event.target.value;
    syncChatModelInfo(state.chat.selectedModel).catch((error) => {
      setStatus("Chat model error", error.message);
    });
    renderChatStatusLine();
  });
  el("#chatForm").addEventListener("submit", (event) => {
    submitChat(event).catch((error) => {
      state.chat.loading = false;
      updateChatControls();
      state.chat.error = error.message;
      state.chat.messages.push({
        role: "system",
        content: `Chat error: ${error.message}`,
      });
      renderChatMessages();
      renderChatStatusLine();
    });
  });
  el("#chatInput").addEventListener("input", () => {
    renderChatStatusLine();
  });
  el("#chatInput").addEventListener("keydown", handleChatComposerKeydown);
}

async function init() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  buildHeroStats(state.config);
  setStatus("Ready", "Upload a farm folder to see the yield estimate and crop drivers.");
  wireTabs();
  wireChat();
  loadChatModels().catch((error) => {
    state.chat.error = error.message;
    renderChatStatusLine();
  });
  el("#uploadForm").addEventListener("submit", (event) => {
    submitUpload(event).catch((error) => setStatus("Failed", error.message));
  });
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => setStatus("Failed to load", error.message));
});
