/* ==========================================================
   IAIFI Research Landscape — scatter.js
   Canvas 2D renderer. Zero external dependencies.
   Dark constellation theme.
   ========================================================== */

(function () {
  "use strict";

  // ── Constants ──────────────────────────────────────────────
  var COLOR_AI         = "#60a5fa";
  var COLOR_PHYSICS    = "#fbbf24";
  var COLOR_BOTH       = "#a78bfa";
  var COLOR_BG         = "rgba(148, 163, 184, 0.12)";
  var COLOR_NOISE      = "rgba(100, 116, 139, 0.2)";

  var SIZE_BG    = 1.0;
  var SIZE_IAIFI = 4;

  var SEARCH_DEBOUNCE_MS = 200;
  var YEAR_DEBOUNCE_MS   = 150;

  // Category palette (hex strings for Canvas)
  var CATEGORY_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#b3b3b3", "#996633", "#4db34d", "#7030a0",
  ];

  // Spatial index grid resolution
  var GRID_CELLS = 128;

  // Hover hit radius in screen pixels
  var HIT_RADIUS = 12;

  // ── State ──────────────────────────────────────────────────
  var allPapers       = [];
  var paperIndex      = {};   // id -> paper
  var paperIdxMap     = {};   // id -> array index
  var clusters        = [];
  var clusterMap      = {};   // cluster id -> cluster
  var abstracts       = {};   // id -> abstract text
  var meta            = {};

  var activeTheme     = "all";
  var colorMode       = "theme";
  var searchQuery     = "";
  var yearMin         = 0;
  var yearMax         = 9999;
  var dataYearMin     = 0;
  var dataYearMax     = 9999;
  var citationSizing  = false;
  var hasCitations    = false;
  var hoverAllPapers  = false;

  var panelPaperId      = null;
  var panelNeighborPage = 0;

  var categoryColorMap  = {};
  var clusterLabelEls   = [];

  // IAIFI bounding box (computed once)
  var iaifiBounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };

  // Canvas + transform state
  var canvas, ctx;
  var transform = { x: 0, y: 0, scale: 1 };
  var dataBounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };

  // Spatial index
  var grid = null;
  var gridCellW = 0;
  var gridCellH = 0;

  // Interaction state
  var isDragging = false;
  var dragStart  = { x: 0, y: 0 };
  var dragTransformStart = { x: 0, y: 0 };
  var hoveredIdx = -1;
  var selectedIdx = -1;

  // Animation frame tracking
  var needsRedraw = true;
  var rafId = null;

  // Cluster label visibility cache
  var sortedClusters = [];

  // ── DOM refs ──────────────────────────────────────────────
  var elContainer, elTooltip, elPanel, elClosePanel, elPanelTitle,
      elPanelAuthors, elPanelMeta, elPanelAbstract, elPanelNeighbors,
      elShowMore, elPanelArxiv, elSearch, elSearchClear, elColorMode,
      elYearMin, elYearMax, elYearRange, elDataVersion, elCitToggle,
      elCitLegend, elCitWrap, elHoverAllToggle, elPaperCount,
      elSearchCount, elPanelBgNotice;

  function cacheDom() {
    var q = function (s) { return document.querySelector(s); };
    elContainer      = q("#scatter-container");
    elTooltip        = q("#tooltip");
    elPanel          = q("#side-panel");
    elClosePanel     = q("#close-panel");
    elPanelTitle     = q("#panel-title");
    elPanelAuthors   = q("#panel-authors");
    elPanelMeta      = q("#panel-meta");
    elPanelAbstract  = q("#panel-abstract");
    elPanelNeighbors = q("#panel-neighbors");
    elShowMore       = q("#show-more-neighbors");
    elPanelArxiv     = q("#panel-arxiv");
    elSearch         = q("#search");
    elSearchClear    = q("#search-clear");
    elColorMode      = q("#color-mode");
    elYearMin        = q("#year-min");
    elYearMax        = q("#year-max");
    elYearRange      = q("#year-range");
    elDataVersion    = q("#data-version");
    elCitToggle      = q("#citation-toggle");
    elCitLegend      = q("#citation-legend");
    elCitWrap        = q("#citation-toggle-wrap");
    elHoverAllToggle = q("#hover-all-toggle");
    elPaperCount     = q("#paper-count");
    elSearchCount    = q("#search-count");
    elPanelBgNotice  = q("#panel-bg-notice");
  }

  // ── Utility ────────────────────────────────────────────────
  function debounce(fn, ms) {
    var timer;
    return function () {
      var self = this, args = arguments;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(self, args); }, ms);
    };
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function escapeHtml(s) {
    if (s == null) return "";
    var div = document.createElement("div");
    div.textContent = String(s);
    return div.innerHTML;
  }

  // ── Coordinate transforms ─────────────────────────────────
  function dataToScreenX(dx) {
    return (dx - transform.x) * transform.scale;
  }

  function dataToScreenY(dy) {
    var h = canvas._logicalH || elContainer.clientHeight;
    return h - (dy - transform.y) * transform.scale;
  }

  function screenToDataX(sx) {
    return sx / transform.scale + transform.x;
  }

  function screenToDataY(sy) {
    var h = canvas._logicalH || elContainer.clientHeight;
    return (h - sy) / transform.scale + transform.y;
  }

  // Fit view to IAIFI papers with padding
  function fitToIaifi() {
    var pad = 0.15; // 15% padding
    var dw = iaifiBounds.xMax - iaifiBounds.xMin;
    var dh = iaifiBounds.yMax - iaifiBounds.yMin;
    if (dw === 0) dw = 1;
    if (dh === 0) dh = 1;

    var cw = canvas._logicalW || elContainer.clientWidth;
    var ch = canvas._logicalH || elContainer.clientHeight;

    var scaleX = cw / (dw * (1 + 2 * pad));
    var scaleY = ch / (dh * (1 + 2 * pad));
    transform.scale = Math.min(scaleX, scaleY);

    var visW = cw / transform.scale;
    var visH = ch / transform.scale;
    transform.x = iaifiBounds.xMin - (visW - dw) / 2;
    transform.y = iaifiBounds.yMin - (visH - dh) / 2;
  }

  // Initialize transform to fit all data with padding (fallback)
  function fitToData() {
    var pad = 0.05;
    var dw = dataBounds.xMax - dataBounds.xMin;
    var dh = dataBounds.yMax - dataBounds.yMin;
    if (dw === 0) dw = 1;
    if (dh === 0) dh = 1;

    var cw = canvas._logicalW || elContainer.clientWidth;
    var ch = canvas._logicalH || elContainer.clientHeight;

    var scaleX = cw / (dw * (1 + 2 * pad));
    var scaleY = ch / (dh * (1 + 2 * pad));
    transform.scale = Math.min(scaleX, scaleY);

    var visW = cw / transform.scale;
    var visH = ch / transform.scale;
    transform.x = dataBounds.xMin - (visW - dw) / 2;
    transform.y = dataBounds.yMin - (visH - dh) / 2;
  }

  // ── Color helpers ──────────────────────────────────────────
  function rgbaStr(r, g, b, a) {
    return "rgba(" + Math.round(r * 255) + "," + Math.round(g * 255) + "," + Math.round(b * 255) + "," + a + ")";
  }

  // Parse hex color to {r, g, b} in 0-255
  function hexToRgb(hex) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return { r: r, g: g, b: b };
  }

  function getThemeColor(p) {
    if (!p.iaifi) {
      return COLOR_BG;
    }
    switch (p.theme) {
      case "AI":      return COLOR_AI;
      case "Physics": return COLOR_PHYSICS;
      case "Both":    return COLOR_BOTH;
      default:        return COLOR_AI;
    }
  }

  function getThemeColorWithOpacity(p) {
    if (!p.iaifi) {
      return COLOR_BG;
    }
    // IAIFI papers in noise cluster: slightly reduced opacity
    if (p.cluster === -1 || p.cluster == null) {
      switch (p.theme) {
        case "AI":      return "rgba(96,165,250,0.8)";
        case "Physics": return "rgba(251,191,36,0.8)";
        case "Both":    return "rgba(167,139,250,0.8)";
        default:        return "rgba(96,165,250,0.8)";
      }
    }
    // Full opacity IAIFI papers
    switch (p.theme) {
      case "AI":      return "rgba(96,165,250,0.9)";
      case "Physics": return "rgba(251,191,36,0.9)";
      case "Both":    return "rgba(167,139,250,0.9)";
      default:        return "rgba(96,165,250,0.9)";
    }
  }

  // Return the hex base color for glow effects (IAIFI papers only)
  function getThemeHex(p) {
    switch (p.theme) {
      case "AI":      return COLOR_AI;
      case "Physics": return COLOR_PHYSICS;
      case "Both":    return COLOR_BOTH;
      default:        return COLOR_AI;
    }
  }

  function getCategoryColor(p) {
    var cat = (p.cat && p.cat.length > 0) ? p.cat[0] : "other";
    if (!p.iaifi) return "rgba(148, 163, 184, 0.12)";
    return categoryColorMap[cat] || COLOR_NOISE;
  }

  function getEnrichmentColor(p) {
    if (!p.iaifi) return "rgba(148, 163, 184, 0.12)";
    if (p.cluster == null || p.cluster === -1) return COLOR_NOISE;
    var cl = clusterMap[p.cluster];
    if (!cl || !cl.enrichment) return COLOR_NOISE;

    var e = cl.enrichment;
    if (e.ci_lo == null || e.ci_hi == null || (e.ci_lo <= 0 && e.ci_hi >= 0)) {
      return COLOR_NOISE;
    }

    var val = e.log2 || 0;
    var t = clamp(val / 3, -1, 1);

    if (t >= 0) {
      return rgbaStr(1.0, 1.0 - t * 0.7, 1.0 - t * 0.85, 0.9);
    } else {
      var s = -t;
      return rgbaStr(1.0 - s * 0.87, 1.0 - s * 0.42, 1.0, 0.9);
    }
  }

  function getPointColor(p) {
    if (colorMode === "category")   return getCategoryColor(p);
    if (colorMode === "enrichment") return getEnrichmentColor(p);
    return getThemeColorWithOpacity(p);
  }

  // Dimmed background color when theme filter is active
  function getDimmedBgColor() {
    return "rgba(100, 116, 139, 0.04)";
  }

  function getPointSize(p) {
    if (citationSizing && p.iaifi && p.cit != null) {
      return clamp(3 + Math.log2(p.cit + 1), 3, 12);
    }
    return p.iaifi ? SIZE_IAIFI : SIZE_BG;
  }

  // ── Filter logic ───────────────────────────────────────────
  function passesFilters(p) {
    if (activeTheme !== "all" && p.iaifi && p.theme !== activeTheme) return false;
    if (p.yr != null && (p.yr < yearMin || p.yr > yearMax)) return false;
    if (searchQuery) {
      var q = searchQuery.toLowerCase();
      var title   = (p.t || "").toLowerCase();
      var authors = (p.a || "").toLowerCase();
      if (title.indexOf(q) === -1 && authors.indexOf(q) === -1) return false;
    }
    return true;
  }

  // ── Paper count ────────────────────────────────────────────
  function updatePaperCount() {
    if (!elPaperCount) return;
    var nIaifi = 0;
    var nBg = 0;
    var nMatch = 0;
    var hasFilter = activeTheme !== "all" || searchQuery || yearMin > dataYearMin || yearMax < dataYearMax;

    for (var i = 0; i < allPapers.length; i++) {
      var p = allPapers[i];
      if (p.iaifi) nIaifi++;
      else nBg++;
      if (hasFilter && passesFilters(p)) nMatch++;
    }

    if (hasFilter) {
      elPaperCount.textContent = "Showing " + nMatch.toLocaleString() + " matching papers of " +
        nIaifi.toLocaleString() + " IAIFI + " + nBg.toLocaleString() + " background";
    } else {
      elPaperCount.textContent = "Showing " + nIaifi.toLocaleString() + " IAIFI papers + " +
        nBg.toLocaleString() + " background";
    }

    // Search count
    if (elSearchCount) {
      if (searchQuery) {
        var sc = 0;
        for (var j = 0; j < allPapers.length; j++) {
          if (passesFilters(allPapers[j])) sc++;
        }
        elSearchCount.textContent = sc + " matches";
        elSearchCount.classList.remove("hidden");
      } else {
        elSearchCount.classList.add("hidden");
      }
    }
  }

  // ── Spatial Index ──────────────────────────────────────────
  function buildSpatialIndex() {
    var dw = dataBounds.xMax - dataBounds.xMin;
    var dh = dataBounds.yMax - dataBounds.yMin;
    if (dw === 0) dw = 1;
    if (dh === 0) dh = 1;

    gridCellW = dw / GRID_CELLS;
    gridCellH = dh / GRID_CELLS;

    grid = new Array(GRID_CELLS * GRID_CELLS);
    for (var i = 0; i < grid.length; i++) grid[i] = null;

    for (var idx = 0; idx < allPapers.length; idx++) {
      var p = allPapers[idx];
      var ci = Math.floor((p.x - dataBounds.xMin) / gridCellW);
      var cj = Math.floor((p.y - dataBounds.yMin) / gridCellH);
      ci = clamp(ci, 0, GRID_CELLS - 1);
      cj = clamp(cj, 0, GRID_CELLS - 1);
      var key = cj * GRID_CELLS + ci;
      if (grid[key] === null) grid[key] = [];
      grid[key].push(idx);
    }
  }

  function findNearestPoint(sx, sy, maxScreenDist, iaifiOnly) {
    if (!grid) return -1;

    var dx = screenToDataX(sx);
    var dy = screenToDataY(sy);

    var dataRadius = maxScreenDist / transform.scale;

    var ciMin = Math.floor((dx - dataRadius - dataBounds.xMin) / gridCellW);
    var ciMax = Math.floor((dx + dataRadius - dataBounds.xMin) / gridCellW);
    var cjMin = Math.floor((dy - dataRadius - dataBounds.yMin) / gridCellH);
    var cjMax = Math.floor((dy + dataRadius - dataBounds.yMin) / gridCellH);

    ciMin = clamp(ciMin, 0, GRID_CELLS - 1);
    ciMax = clamp(ciMax, 0, GRID_CELLS - 1);
    cjMin = clamp(cjMin, 0, GRID_CELLS - 1);
    cjMax = clamp(cjMax, 0, GRID_CELLS - 1);

    var bestIdx = -1;
    var bestDist = maxScreenDist * maxScreenDist;

    for (var cj = cjMin; cj <= cjMax; cj++) {
      for (var ci = ciMin; ci <= ciMax; ci++) {
        var cell = grid[cj * GRID_CELLS + ci];
        if (!cell) continue;
        for (var k = 0; k < cell.length; k++) {
          var idx = cell[k];
          var p = allPapers[idx];
          // Skip non-IAIFI if iaifiOnly
          if (iaifiOnly && !p.iaifi) continue;
          var psx = dataToScreenX(p.x);
          var psy = dataToScreenY(p.y);
          var ddx = psx - sx;
          var ddy = psy - sy;
          var d2 = ddx * ddx + ddy * ddy;
          if (d2 < bestDist) {
            bestDist = d2;
            bestIdx = idx;
          }
        }
      }
    }

    return bestIdx;
  }

  // ── Canvas Drawing ─────────────────────────────────────────
  function resizeCanvas() {
    var dpr = window.devicePixelRatio || 1;
    var w = elContainer.clientWidth;
    var h = elContainer.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvas._logicalW = w;
    canvas._logicalH = h;
  }

  function drawFrame() {
    if (!needsRedraw) {
      rafId = requestAnimationFrame(drawFrame);
      return;
    }
    needsRedraw = false;

    var w = canvas._logicalW || elContainer.clientWidth;
    var h = canvas._logicalH || elContainer.clientHeight;

    ctx.clearRect(0, 0, w, h);

    // Dark radial gradient background (constellation map feel)
    var cx = w / 2;
    var cy = h / 2;
    var maxR = Math.max(w, h) * 0.7;
    var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
    grad.addColorStop(0, "#131b2e");
    grad.addColorStop(1, "#0a0f1a");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    var n = allPapers.length;
    if (n === 0) {
      rafId = requestAnimationFrame(drawFrame);
      return;
    }

    var hasFilter = activeTheme !== "all" || searchQuery || yearMin > dataYearMin || yearMax < dataYearMax;
    var themeFilterActive = activeTheme !== "all";

    var TWO_PI = Math.PI * 2;

    // We use layers: bg dimmed, bg normal, IAIFI glow, IAIFI solid
    var dimBatch = [];       // dimmed non-matching
    var bgBatch = [];        // background papers (normal)
    var matchBatches = {};   // IAIFI matching, keyed by color
    // For glow: collect IAIFI points grouped by their hex base color
    var glowBatches = {};    // hex color -> [sx, sy, r, ...]

    for (var i = 0; i < n; i++) {
      var p = allPapers[i];
      var sx = dataToScreenX(p.x);
      var sy = dataToScreenY(p.y);

      // Cull off-screen
      if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) continue;

      var baseSize = getPointSize(p);
      var r = baseSize * Math.min(transform.scale / 40, 2);
      r = clamp(r, 0.5, 20);

      var passes = passesFilters(p);

      if (!p.iaifi) {
        // Background paper
        if (hasFilter && !passes) {
          dimBatch.push(sx, sy, r);
        } else if (themeFilterActive) {
          dimBatch.push(sx, sy, r);
        } else {
          bgBatch.push(sx, sy, r);
        }
      } else {
        // IAIFI paper
        if (hasFilter && !passes) {
          dimBatch.push(sx, sy, r);
        } else {
          var color = getPointColor(p);
          if (!matchBatches[color]) matchBatches[color] = [];
          matchBatches[color].push(sx, sy, r);

          // Collect glow data (theme hex for glow pass)
          var glowHex = getThemeHex(p);
          if (!glowBatches[glowHex]) glowBatches[glowHex] = [];
          glowBatches[glowHex].push(sx, sy, r);
        }
      }
    }

    // Pass 1: Dimmed points (barely visible dust)
    if (dimBatch.length > 0) {
      ctx.fillStyle = "rgba(100, 116, 139, 0.04)";
      ctx.beginPath();
      for (var j = 0; j < dimBatch.length; j += 3) {
        ctx.moveTo(dimBatch[j] + dimBatch[j + 2], dimBatch[j + 1]);
        ctx.arc(dimBatch[j], dimBatch[j + 1], dimBatch[j + 2], 0, TWO_PI);
      }
      ctx.fill();
    }

    // Pass 2: Normal background points (distant stars)
    if (bgBatch.length > 0) {
      ctx.fillStyle = COLOR_BG;
      ctx.beginPath();
      for (var j = 0; j < bgBatch.length; j += 3) {
        ctx.moveTo(bgBatch[j] + bgBatch[j + 2], bgBatch[j + 1]);
        ctx.arc(bgBatch[j], bgBatch[j + 1], bgBatch[j + 2], 0, TWO_PI);
      }
      ctx.fill();
    }

    // Pass 3: IAIFI glow halos (draw before solid dots)
    var glowColors = Object.keys(glowBatches);
    for (var gi = 0; gi < glowColors.length; gi++) {
      var ghex = glowColors[gi];
      var gpts = glowBatches[ghex];
      var rgb = hexToRgb(ghex);
      ctx.fillStyle = "rgba(" + rgb.r + "," + rgb.g + "," + rgb.b + ",0.12)";
      ctx.beginPath();
      for (var j = 0; j < gpts.length; j += 3) {
        var glowR = gpts[j + 2] * 2.5;
        ctx.moveTo(gpts[j] + glowR, gpts[j + 1]);
        ctx.arc(gpts[j], gpts[j + 1], glowR, 0, TWO_PI);
      }
      ctx.fill();
    }

    // Pass 4: IAIFI solid dots (on top)
    var colors = Object.keys(matchBatches);
    for (var ci = 0; ci < colors.length; ci++) {
      var c = colors[ci];
      var pts = matchBatches[c];
      ctx.fillStyle = c;
      ctx.beginPath();
      for (var j = 0; j < pts.length; j += 3) {
        ctx.moveTo(pts[j] + pts[j + 2], pts[j + 1]);
        ctx.arc(pts[j], pts[j + 1], pts[j + 2], 0, TWO_PI);
      }
      ctx.fill();
    }

    // Hovered point: draw glowing ring
    if (hoveredIdx >= 0 && hoveredIdx < n) {
      var hp = allPapers[hoveredIdx];
      var hsx = dataToScreenX(hp.x);
      var hsy = dataToScreenY(hp.y);
      var hr = 7;

      // Outer glow (3 concentric rings with decreasing opacity)
      ctx.globalAlpha = 0.08;
      ctx.beginPath();
      ctx.arc(hsx, hsy, hr + 12, 0, TWO_PI);
      ctx.fillStyle = "#fff";
      ctx.fill();

      ctx.globalAlpha = 0.12;
      ctx.beginPath();
      ctx.arc(hsx, hsy, hr + 8, 0, TWO_PI);
      ctx.fill();

      ctx.globalAlpha = 0.2;
      ctx.beginPath();
      ctx.arc(hsx, hsy, hr + 4, 0, TWO_PI);
      ctx.fill();

      ctx.globalAlpha = 1.0;

      // White ring
      ctx.beginPath();
      ctx.arc(hsx, hsy, hr + 2, 0, TWO_PI);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.stroke();

      // Colored fill
      ctx.beginPath();
      ctx.arc(hsx, hsy, hr, 0, TWO_PI);
      ctx.fillStyle = hp.iaifi ? getPointColor(hp) : "#64748b";
      ctx.fill();
    }

    // Selected point: bright white ring + glow
    if (selectedIdx >= 0 && selectedIdx < n && selectedIdx !== hoveredIdx) {
      var sp = allPapers[selectedIdx];
      var ssx = dataToScreenX(sp.x);
      var ssy = dataToScreenY(sp.y);
      var sr = 7;

      // Strong glow
      ctx.globalAlpha = 0.15;
      ctx.beginPath();
      ctx.arc(ssx, ssy, sr + 14, 0, TWO_PI);
      ctx.fillStyle = "#fff";
      ctx.fill();

      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.arc(ssx, ssy, sr + 8, 0, TWO_PI);
      ctx.fill();

      ctx.globalAlpha = 1.0;

      // Outer ring
      ctx.beginPath();
      ctx.arc(ssx, ssy, sr + 4, 0, TWO_PI);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.stroke();

      // Inner bright ring
      ctx.beginPath();
      ctx.arc(ssx, ssy, sr + 2, 0, TWO_PI);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff";
      ctx.stroke();

      // Colored fill
      ctx.beginPath();
      ctx.arc(ssx, ssy, sr, 0, TWO_PI);
      ctx.fillStyle = sp.iaifi ? getPointColor(sp) : "#64748b";
      ctx.fill();
    }

    // Cluster labels
    drawClusterLabels(w, h);

    rafId = requestAnimationFrame(drawFrame);
  }

  function drawClusterLabels(w, h) {
    if (!sortedClusters || sortedClusters.length === 0) return;

    // Compute zoom level relative to IAIFI fit
    var dw = iaifiBounds.xMax - iaifiBounds.xMin;
    if (dw === 0) dw = 1;
    var baseScale = w / dw;
    var zoomFactor = transform.scale / baseScale;

    if (zoomFactor < 0.2 || zoomFactor > 25) return;

    var fontSize = clamp(11 / Math.sqrt(zoomFactor), 9, 18);
    ctx.font = "600 " + fontSize + "px \"Inter\", -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Determine how many labels to show based on zoom
    // With 200+ clusters, be conservative at low zoom and reveal progressively
    var maxLabels;
    if (zoomFactor < 0.5) {
      maxLabels = 3;
    } else if (zoomFactor < 0.8) {
      maxLabels = 5;
    } else if (zoomFactor < 1.5) {
      maxLabels = 8;
    } else if (zoomFactor < 3.0) {
      maxLabels = 15;
    } else if (zoomFactor < 6.0) {
      maxLabels = 25;
    } else {
      maxLabels = sortedClusters.length;
    }

    // Collision margin: extra pixels around each label to prevent near-overlaps
    var collisionMargin = zoomFactor < 2.0 ? 12 : 6;

    // Collision detection: track placed label bounding boxes
    var placed = [];

    for (var i = 0; i < sortedClusters.length && placed.length < maxLabels; i++) {
      var cl = sortedClusters[i];
      if (cl.cx == null || cl.cy == null) continue;
      var label = cl.label || cl.auto_label || "";
      if (!label) continue;

      var sx = dataToScreenX(cl.cx);
      var sy = dataToScreenY(cl.cy);

      // Skip off-screen
      if (sx < -100 || sx > w + 100 || sy < -50 || sy > h + 50) continue;

      // Measure text for collision (with margin for breathing room)
      var tw = ctx.measureText(label).width + 16; // pill padding
      var th = fontSize + 10;
      var lx = sx - tw / 2;
      var ly = sy - th / 2;

      // Expanded bounding box for collision test (adds margin on all sides)
      var cx0 = lx - collisionMargin;
      var cy0 = ly - collisionMargin;
      var cw  = tw + collisionMargin * 2;
      var ch  = th + collisionMargin * 2;

      // Check overlap with already placed labels
      var overlaps = false;
      for (var j = 0; j < placed.length; j++) {
        var pb = placed[j];
        if (cx0 < pb.x + pb.w && cx0 + cw > pb.x &&
            cy0 < pb.y + pb.h && cy0 + ch > pb.y) {
          overlaps = true;
          break;
        }
      }
      if (overlaps) continue;

      placed.push({ x: cx0, y: cy0, w: cw, h: ch });

      // Draw dark pill background with slight transparency
      var pillR = th / 2;
      ctx.fillStyle = "rgba(13, 17, 23, 0.7)";
      ctx.beginPath();
      ctx.moveTo(lx + pillR, ly);
      ctx.lineTo(lx + tw - pillR, ly);
      ctx.arc(lx + tw - pillR, ly + th / 2, pillR, -Math.PI / 2, Math.PI / 2);
      ctx.lineTo(lx + pillR, ly + th);
      ctx.arc(lx + pillR, ly + th / 2, pillR, Math.PI / 2, -Math.PI / 2);
      ctx.closePath();
      ctx.fill();

      // Subtle glowing border
      ctx.strokeStyle = "rgba(56, 189, 248, 0.12)";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Label text - light with subtle glow
      ctx.save();
      ctx.shadowColor = "rgba(56, 189, 248, 0.25)";
      ctx.shadowBlur = 6;
      ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
      ctx.fillText(label, sx, sy);
      ctx.restore();
    }
  }

  function requestRedraw() {
    needsRedraw = true;
  }

  // ── Zoom & Pan ─────────────────────────────────────────────
  function onWheel(e) {
    e.preventDefault();

    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    var dx = screenToDataX(mx);
    var dy = screenToDataY(my);

    var factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    var newScale = transform.scale * factor;
    newScale = clamp(newScale, 5, 50000);
    transform.scale = newScale;

    var h = canvas._logicalH || elContainer.clientHeight;
    transform.x = dx - mx / transform.scale;
    transform.y = dy - (h - my) / transform.scale;

    requestRedraw();
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    isDragging = true;
    dragStart.x = e.clientX;
    dragStart.y = e.clientY;
    dragTransformStart.x = transform.x;
    dragTransformStart.y = transform.y;
    canvas.style.cursor = "grabbing";
  }

  function onMouseMove(e) {
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    if (isDragging) {
      var ddx = e.clientX - dragStart.x;
      var ddy = e.clientY - dragStart.y;
      transform.x = dragTransformStart.x - ddx / transform.scale;
      transform.y = dragTransformStart.y + ddy / transform.scale;
      requestRedraw();
      hideTooltip();
      return;
    }

    // Hover hit test: IAIFI-first unless hoverAllPapers is on
    var iaifiOnly = !hoverAllPapers;
    var idx = findNearestPoint(mx, my, HIT_RADIUS, iaifiOnly);

    // If IAIFI-only found nothing and hoverAll is off, also try a closer range for IAIFI
    // (don't fall through to background)

    if (idx !== hoveredIdx) {
      hoveredIdx = idx;
      if (idx >= 0) {
        showTooltip(idx, e);
        canvas.style.cursor = "pointer";
      } else {
        hideTooltip();
        canvas.style.cursor = "grab";
      }
      requestRedraw();
    } else if (idx >= 0) {
      positionTooltip(e);
    }
  }

  function onMouseUp(e) {
    if (!isDragging) return;
    var ddx = e.clientX - dragStart.x;
    var ddy = e.clientY - dragStart.y;
    var moved = Math.abs(ddx) + Math.abs(ddy) > 3;

    isDragging = false;
    canvas.style.cursor = "grab";

    if (!moved) {
      var rect = canvas.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      // Click: try IAIFI first, then all
      var idx = findNearestPoint(mx, my, HIT_RADIUS, !hoverAllPapers);
      if (idx < 0 && !hoverAllPapers) {
        idx = findNearestPoint(mx, my, HIT_RADIUS, false);
      }
      if (idx >= 0) {
        selectedIdx = idx;
        openPanel(allPapers[idx].id);
        requestRedraw();
      }
    }
  }

  function onMouseLeave() {
    if (isDragging) {
      isDragging = false;
      canvas.style.cursor = "grab";
    }
    hoveredIdx = -1;
    hideTooltip();
    requestRedraw();
  }

  // ── Tooltip ────────────────────────────────────────────────
  function showTooltip(idx, e) {
    var p = allPapers[idx];
    if (!p) return;

    var cat = (p.cat && p.cat.length > 0) ? p.cat[0] : "";

    if (p.iaifi) {
      // Rich tooltip for IAIFI papers
      var authors = p.a || "";
      if (authors.length > 80) authors = authors.substring(0, 77) + "...";

      var themeBadgeColor = "";
      switch (p.theme) {
        case "AI": themeBadgeColor = COLOR_AI; break;
        case "Physics": themeBadgeColor = COLOR_PHYSICS; break;
        case "Both": themeBadgeColor = COLOR_BOTH; break;
      }

      var html = '<div class="tooltip-title">' + escapeHtml(p.t || "Untitled") + '</div>';
      html += '<div class="tooltip-authors">' + escapeHtml(authors) + '</div>';
      html += '<div class="tooltip-meta">' + escapeHtml(p.yr + " \u00B7 " + cat);
      if (p.theme) {
        html += '<span class="tooltip-theme-badge" style="background:' + themeBadgeColor + '">' +
          escapeHtml(p.theme) + '</span>';
      }
      html += '</div>';
      elTooltip.innerHTML = html;
    } else {
      // Minimal tooltip for background papers
      elTooltip.innerHTML =
        '<div class="tooltip-title">' + escapeHtml(p.t || "Untitled") + '</div>' +
        '<div class="tooltip-meta">' + escapeHtml(p.yr + " \u00B7 " + cat) + '</div>' +
        '<div class="tooltip-bg-label">(background)</div>';
    }

    elTooltip.classList.remove("hidden");
    if (e) positionTooltip(e);
  }

  function positionTooltip(e) {
    var rect = elContainer.getBoundingClientRect();
    var left = e.clientX - rect.left + 12;
    var top  = e.clientY - rect.top  + 12;

    var tw = elTooltip.offsetWidth;
    var th = elTooltip.offsetHeight;
    if (left + tw > rect.width)  left = left - tw - 24;
    if (top  + th > rect.height) top  = top  - th - 24;

    elTooltip.style.left = left + "px";
    elTooltip.style.top  = top  + "px";
  }

  function hideTooltip() {
    elTooltip.classList.add("hidden");
  }

  // ── Side Panel ─────────────────────────────────────────────
  function openPanel(paperId) {
    var p = paperIndex[paperId];
    if (!p) return;

    panelPaperId = paperId;
    panelNeighborPage = 0;

    // Show/hide background notice
    if (elPanelBgNotice) {
      if (p.iaifi) {
        elPanelBgNotice.classList.add("hidden");
      } else {
        elPanelBgNotice.classList.remove("hidden");
      }
    }

    elPanelTitle.textContent   = p.t || "Untitled";
    elPanelAuthors.textContent = p.a || "";

    var html = "";
    if (p.yr) html += '<span class="meta-tag">' + p.yr + '</span>';
    if (p.cat) {
      p.cat.forEach(function (c) {
        html += '<span class="meta-tag">' + escapeHtml(c) + '</span>';
      });
    }
    if (p.iaifi && p.theme) {
      html += '<span class="meta-tag">IAIFI: ' + escapeHtml(p.theme) + '</span>';
    }
    if (p.cit != null) {
      html += '<span class="meta-tag">' + p.cit + ' citations</span>';
    }
    elPanelMeta.innerHTML = html;

    var abs = abstracts[paperId] || p.abstract || "";
    elPanelAbstract.textContent = abs || "(Abstract not available)";

    elPanelArxiv.href = "https://arxiv.org/abs/" + paperId;

    renderNeighbors();

    elPanel.classList.remove("hidden");
    void elPanel.offsetHeight;
    elPanel.classList.add("visible");
  }

  function closePanel() {
    elPanel.classList.remove("visible");
    setTimeout(function () {
      elPanel.classList.add("hidden");
    }, 300);
    panelPaperId = null;
    selectedIdx = -1;
    requestRedraw();
  }

  function renderNeighbors() {
    var p = paperIndex[panelPaperId];
    if (!p || !p.nn || p.nn.length === 0) {
      elPanelNeighbors.innerHTML = '<li>No neighbors available</li>';
      elShowMore.classList.add("hidden");
      return;
    }

    // Sort neighbors: IAIFI papers first, then background
    var iaifiNeighbors = [];
    var bgNeighbors = [];
    for (var i = 0; i < p.nn.length; i++) {
      var nid = p.nn[i];
      var np = paperIndex[nid];
      if (np && np.iaifi) {
        iaifiNeighbors.push(nid);
      } else {
        bgNeighbors.push(nid);
      }
    }
    var sortedNn = iaifiNeighbors.concat(bgNeighbors);

    var end = Math.min((panelNeighborPage + 1) * 5, sortedNn.length);
    var visible = sortedNn.slice(0, end);

    elPanelNeighbors.innerHTML = "";
    visible.forEach(function (nid) {
      var np = paperIndex[nid];
      var li = document.createElement("li");

      if (np) {
        var cat = (np.cat && np.cat.length > 0) ? np.cat[0] : "";
        var badge = np.iaifi ? '<span class="neighbor-iaifi-badge">IAIFI</span>' : '';
        li.innerHTML =
          '<div class="neighbor-title">' + escapeHtml(np.t || nid) + badge + '</div>' +
          '<div class="neighbor-meta">' + escapeHtml((np.yr || "") + " \u00B7 " + cat) + '</div>';
        li.addEventListener("click", function () {
          openPanel(nid);
          highlightPoint(nid);
        });
      } else {
        li.innerHTML = '<div class="neighbor-title">' + escapeHtml(nid) + '</div>';
        li.addEventListener("click", function () {
          window.open("https://arxiv.org/abs/" + nid, "_blank");
        });
      }
      elPanelNeighbors.appendChild(li);
    });

    if (end < sortedNn.length) {
      elShowMore.classList.remove("hidden");
    } else {
      elShowMore.classList.add("hidden");
    }
  }

  function highlightPoint(paperId) {
    var idx = paperIdxMap[paperId];
    if (idx != null) {
      selectedIdx = idx;
      requestRedraw();
    }
  }

  // ── Year Slider ────────────────────────────────────────────
  function initSliders() {
    elYearMin.min   = dataYearMin;
    elYearMin.max   = dataYearMax;
    elYearMin.value = dataYearMin;
    elYearMin.step  = 1;

    elYearMax.min   = dataYearMin;
    elYearMax.max   = dataYearMax;
    elYearMax.value = dataYearMax;
    elYearMax.step  = 1;

    updateYearLabel();

    var onSliderMin = debounce(function () {
      var lo = parseInt(elYearMin.value, 10);
      var hi = parseInt(elYearMax.value, 10);
      if (lo > hi) { lo = hi; elYearMin.value = lo; }
      yearMin = lo;
      updateYearLabel();
      applyFilters();
    }, YEAR_DEBOUNCE_MS);

    var onSliderMax = debounce(function () {
      var lo = parseInt(elYearMin.value, 10);
      var hi = parseInt(elYearMax.value, 10);
      if (hi < lo) { hi = lo; elYearMax.value = hi; }
      yearMin = lo;
      yearMax = hi;
      updateYearLabel();
      applyFilters();
    }, YEAR_DEBOUNCE_MS);

    elYearMin.addEventListener("input", onSliderMin);
    elYearMax.addEventListener("input", onSliderMax);
  }

  function updateYearLabel() {
    elYearRange.textContent = yearMin + " \u2013 " + yearMax;
  }

  // ── Search ─────────────────────────────────────────────────
  var onSearchInput = debounce(function () {
    searchQuery = elSearch.value.trim();
    elSearchClear.classList.toggle("hidden", !searchQuery);
    applyFilters();
  }, SEARCH_DEBOUNCE_MS);

  // ── Theme Filters ──────────────────────────────────────────
  function onThemeClick(e) {
    var btn = e.target.closest("button[data-theme]");
    if (!btn) return;
    var all = document.querySelectorAll("#theme-filters button");
    for (var i = 0; i < all.length; i++) all[i].classList.remove("active");
    btn.classList.add("active");
    activeTheme = btn.dataset.theme;
    applyFilters();
  }

  // ── Color Mode ─────────────────────────────────────────────
  function onColorModeChange() {
    colorMode = elColorMode.value;
    applyFilters();
  }

  // ── Citation Toggle ────────────────────────────────────────
  function onCitationToggle() {
    citationSizing = elCitToggle.checked;
    elCitLegend.classList.toggle("hidden", !citationSizing);
    applyFilters();
  }

  // ── Hover All Toggle ──────────────────────────────────────
  function onHoverAllToggle() {
    hoverAllPapers = elHoverAllToggle.checked;
  }

  // ── Apply filters ──────────────────────────────────────────
  function applyFilters() {
    updatePaperCount();
    requestRedraw();
  }

  // ── Event Binding ──────────────────────────────────────────
  function bindEvents() {
    elSearch.addEventListener("input", onSearchInput);
    elSearchClear.addEventListener("click", function () {
      elSearch.value = "";
      searchQuery = "";
      elSearchClear.classList.add("hidden");
      applyFilters();
    });

    document.querySelector("#theme-filters").addEventListener("click", onThemeClick);
    elColorMode.addEventListener("change", onColorModeChange);
    elClosePanel.addEventListener("click", closePanel);

    elShowMore.addEventListener("click", function () {
      panelNeighborPage++;
      renderNeighbors();
    });

    elCitToggle.addEventListener("change", onCitationToggle);

    if (elHoverAllToggle) {
      elHoverAllToggle.addEventListener("change", onHoverAllToggle);
    }

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && elPanel.classList.contains("visible")) {
        closePanel();
      }
    });
  }

  function bindCanvasEvents() {
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseLeave);

    // Touch support for mobile
    var lastTouchDist = 0;
    var lastTouchCenter = { x: 0, y: 0 };

    canvas.addEventListener("touchstart", function (e) {
      e.preventDefault();
      if (e.touches.length === 1) {
        var t = e.touches[0];
        isDragging = true;
        dragStart.x = t.clientX;
        dragStart.y = t.clientY;
        dragTransformStart.x = transform.x;
        dragTransformStart.y = transform.y;
      } else if (e.touches.length === 2) {
        isDragging = false;
        var t0 = e.touches[0], t1 = e.touches[1];
        lastTouchDist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
        lastTouchCenter.x = (t0.clientX + t1.clientX) / 2;
        lastTouchCenter.y = (t0.clientY + t1.clientY) / 2;
      }
    }, { passive: false });

    canvas.addEventListener("touchmove", function (e) {
      e.preventDefault();
      if (e.touches.length === 1 && isDragging) {
        var t = e.touches[0];
        var ddx = t.clientX - dragStart.x;
        var ddy = t.clientY - dragStart.y;
        transform.x = dragTransformStart.x - ddx / transform.scale;
        transform.y = dragTransformStart.y + ddy / transform.scale;
        requestRedraw();
      } else if (e.touches.length === 2) {
        var t0 = e.touches[0], t1 = e.touches[1];
        var dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
        var cx = (t0.clientX + t1.clientX) / 2;
        var cy = (t0.clientY + t1.clientY) / 2;

        if (lastTouchDist > 0) {
          var rect = canvas.getBoundingClientRect();
          var mx = cx - rect.left;
          var my = cy - rect.top;
          var dx = screenToDataX(mx);
          var dy = screenToDataY(my);

          var factor = dist / lastTouchDist;
          transform.scale = clamp(transform.scale * factor, 5, 50000);

          var hh = canvas._logicalH || elContainer.clientHeight;
          transform.x = dx - mx / transform.scale;
          transform.y = dy - (hh - my) / transform.scale;
          requestRedraw();
        }

        lastTouchDist = dist;
        lastTouchCenter.x = cx;
        lastTouchCenter.y = cy;
      }
    }, { passive: false });

    canvas.addEventListener("touchend", function (e) {
      if (e.touches.length === 0) {
        isDragging = false;
        lastTouchDist = 0;
      }
    });

    window.addEventListener("resize", function () {
      var oldW = canvas._logicalW || 1;
      var oldH = canvas._logicalH || 1;
      var centerDx = screenToDataX(oldW / 2);
      var centerDy = screenToDataY(oldH / 2);

      resizeCanvas();

      var newW = canvas._logicalW || 1;
      var newH = canvas._logicalH || 1;
      transform.x = centerDx - newW / (2 * transform.scale);
      transform.y = centerDy - newH / (2 * transform.scale);

      requestRedraw();
    });

    canvas.style.cursor = "grab";
  }

  // ── Scatterplot init ───────────────────────────────────────
  function initCanvas() {
    canvas = document.createElement("canvas");
    elContainer.innerHTML = "";
    elContainer.appendChild(canvas);

    ctx = canvas.getContext("2d");
    resizeCanvas();

    // Fit to IAIFI papers instead of all data
    var hasIaifi = allPapers.some(function (p) { return p.iaifi; });
    if (hasIaifi) {
      fitToIaifi();
    } else {
      fitToData();
    }

    bindCanvasEvents();

    requestRedraw();
    rafId = requestAnimationFrame(drawFrame);
  }

  // ── Main ───────────────────────────────────────────────────
  async function main() {
    cacheDom();
    bindEvents();

    elContainer.innerHTML =
      '<p id="load-status" style="padding:40px;text-align:center;color:#94a3b8;">Loading data...</p>';

    var data;
    try {
      var resp = await fetch("data/papers.json");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      elContainer.innerHTML =
        '<p style="padding:40px;text-align:center;color:#f87171;">' +
        'Error loading data. Make sure <code>data/papers.json</code> exists.<br>' +
        escapeHtml(err.message) + '</p>';
      return;
    }

    meta      = data.meta || {};
    clusters  = data.clusters || [];
    allPapers = data.papers || [];

    // Build cluster lookup
    clusters.forEach(function (c) { clusterMap[c.id] = c; });

    // Build paper indices
    allPapers.forEach(function (p, i) {
      paperIndex[p.id]  = p;
      paperIdxMap[p.id] = i;
    });

    // Compute data bounds (all papers) and IAIFI bounds
    var xMin = Infinity, xMax = -Infinity;
    var yMin = Infinity, yMax = -Infinity;
    var ixMin = Infinity, ixMax = -Infinity;
    var iyMin = Infinity, iyMax = -Infinity;
    var hasIaifiPapers = false;

    for (var i = 0; i < allPapers.length; i++) {
      var p = allPapers[i];
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
      if (p.iaifi) {
        hasIaifiPapers = true;
        if (p.x < ixMin) ixMin = p.x;
        if (p.x > ixMax) ixMax = p.x;
        if (p.y < iyMin) iyMin = p.y;
        if (p.y > iyMax) iyMax = p.y;
      }
    }
    dataBounds.xMin = xMin;
    dataBounds.xMax = xMax;
    dataBounds.yMin = yMin;
    dataBounds.yMax = yMax;

    if (hasIaifiPapers) {
      iaifiBounds.xMin = ixMin;
      iaifiBounds.xMax = ixMax;
      iaifiBounds.yMin = iyMin;
      iaifiBounds.yMax = iyMax;
    } else {
      iaifiBounds = { xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax };
    }

    // Sort clusters by size (largest first) for progressive label reveal
    // Only keep clusters with size >= 40 to avoid label clutter with many clusters
    sortedClusters = clusters.filter(function (c) {
      return (c.size || 0) >= 40;
    }).sort(function (a, b) {
      return (b.size || 0) - (a.size || 0);
    });

    // Citation data detection
    hasCitations = allPapers.some(function (p) {
      return p.cit != null && p.cit > 0;
    });
    if (!hasCitations && elCitWrap) {
      elCitWrap.classList.add("hidden");
    }

    // Year bounds
    var yrs = [];
    allPapers.forEach(function (p) { if (p.yr != null) yrs.push(p.yr); });
    if (yrs.length > 0) {
      dataYearMin = Math.min.apply(null, yrs);
      dataYearMax = Math.max.apply(null, yrs);
    }
    yearMin = dataYearMin;
    yearMax = dataYearMax;

    // Category color map
    var catSet = {};
    allPapers.forEach(function (p) {
      var c = (p.cat && p.cat.length > 0) ? p.cat[0] : "other";
      catSet[c] = true;
    });
    Object.keys(catSet).sort().forEach(function (c, i) {
      categoryColorMap[c] = CATEGORY_PALETTE[i % CATEGORY_PALETTE.length];
    });

    // Display data version in footer
    if (meta.version && elDataVersion) {
      var parts = [];
      if (meta.generated) parts.push(meta.generated.slice(0, 10));
      if (meta.embedding_model) parts.push(meta.embedding_model);
      if (meta.n_total) parts.push(meta.n_total.toLocaleString() + " papers");
      elDataVersion.textContent = parts.join(" | ");
    }

    // Build spatial index
    buildSpatialIndex();

    // Initialize UI
    initSliders();
    updatePaperCount();
    initCanvas();

    // Load abstracts lazily in background
    if (meta.abstracts_split || Object.keys(data.abstracts || {}).length === 0) {
      setTimeout(function () {
        fetch("data/abstracts.json")
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (d) {
            if (d) abstracts = d.abstracts || d;
          })
          .catch(function () { /* graceful degradation */ });
      }, 100);
    } else {
      abstracts = data.abstracts || {};
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { main(); });
  } else {
    main();
  }

})();
