/* ==========================================================
   IAIFI Research Landscape — scatter.js
   Complete frontend: data loading, regl-scatterplot rendering,
   search, filters, color modes, year slider, cluster labels,
   citation sizing, and occlusion diagnostics.
   ========================================================== */

(function () {
  "use strict";

  // ── Constants ──────────────────────────────────────────────
  const COLORS = {
    ai:         [0x21 / 255, 0x96 / 255, 0xF3 / 255, 1.0],   // #2196F3
    physics:    [0xFF / 255, 0x98 / 255, 0x00 / 255, 1.0],   // #FF9800
    both:       [0x9C / 255, 0x27 / 255, 0xB0 / 255, 1.0],   // #9C27B0
    background: [0xe0 / 255, 0xe0 / 255, 0xe0 / 255, 0.3],   // #e0e0e0 30%
    noise:      [0x99 / 255, 0x99 / 255, 0x99 / 255, 0.4],   // grey 40%
  };

  const SIZE_BG    = 3;
  const SIZE_IAIFI = 5;

  const SEARCH_DEBOUNCE_MS = 200;
  const YEAR_DEBOUNCE_MS   = 150;

  // Distinct colors for arXiv categories (up to 20)
  const CATEGORY_PALETTE = [
    [0.12, 0.47, 0.71, 1], [1.00, 0.50, 0.05, 1],
    [0.17, 0.63, 0.17, 1], [0.84, 0.15, 0.16, 1],
    [0.58, 0.40, 0.74, 1], [0.55, 0.34, 0.29, 1],
    [0.89, 0.47, 0.76, 1], [0.50, 0.50, 0.50, 1],
    [0.74, 0.74, 0.13, 1], [0.09, 0.75, 0.81, 1],
    [0.40, 0.76, 0.65, 1], [0.99, 0.55, 0.38, 1],
    [0.55, 0.63, 0.80, 1], [0.91, 0.54, 0.76, 1],
    [0.65, 0.85, 0.33, 1], [1.00, 0.85, 0.56, 1],
    [0.70, 0.70, 0.70, 1], [0.60, 0.44, 0.30, 1],
    [0.30, 0.68, 0.29, 1], [0.44, 0.19, 0.63, 1],
  ];

  // ── State ──────────────────────────────────────────────────
  let scatterplot     = null;
  let allPapers       = [];
  let paperIndex      = {};   // id -> paper object
  let paperIdxMap     = {};   // id -> array index
  let clusters        = [];
  let clusterMap      = {};   // cluster id -> cluster object
  let abstracts       = {};   // id -> abstract text
  let meta            = {};

  let activeTheme     = "all";
  let colorMode       = "theme";
  let searchQuery     = "";
  let yearMin         = 0;
  let yearMax         = 9999;
  let dataYearMin     = 0;
  let dataYearMax     = 9999;
  let citationSizing  = false;
  let hasCitations    = false;

  let panelPaperId      = null;
  let panelNeighborPage = 0;

  let categoryColorMap  = {};
  let clusterLabelEls   = [];

  // ── DOM refs (resolved after DOMContentLoaded) ─────────────
  let elContainer, elTooltip, elPanel, elClosePanel, elPanelTitle,
      elPanelAuthors, elPanelMeta, elPanelAbstract, elPanelNeighbors,
      elShowMore, elPanelArxiv, elSearch, elSearchClear, elColorMode,
      elYearMin, elYearMax, elYearRange, elDataVersion, elCitToggle,
      elCitLegend, elCitWrap;

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

  // ── Color helpers ──────────────────────────────────────────
  function getThemeColor(p) {
    if (!p.iaifi) {
      if (p.cluster === -1 || p.cluster == null) return COLORS.noise;
      return COLORS.background;
    }
    // IAIFI paper base color
    var base;
    switch (p.theme) {
      case "AI":      base = COLORS.ai; break;
      case "Physics": base = COLORS.physics; break;
      case "Both":    base = COLORS.both; break;
      default:        base = COLORS.ai; break;
    }
    // IAIFI papers in noise cluster: keep theme color but reduced opacity
    if (p.cluster === -1 || p.cluster == null) {
      return [base[0], base[1], base[2], 0.7];
    }
    return base;
  }

  function getCategoryColor(p) {
    var cat = (p.cat && p.cat.length > 0) ? p.cat[0] : "other";
    return categoryColorMap[cat] || COLORS.noise;
  }

  function getEnrichmentColor(p) {
    if (p.cluster == null || p.cluster === -1) return COLORS.noise;
    var cl = clusterMap[p.cluster];
    if (!cl || !cl.enrichment) return COLORS.noise;

    var e = cl.enrichment;
    // Grey if either CI bound is null or CI crosses zero
    if (e.ci_lo == null || e.ci_hi == null || (e.ci_lo <= 0 && e.ci_hi >= 0)) {
      return COLORS.noise;
    }

    // Blue-White-Red diverging scale
    var val = e.log2 || 0;
    var t = clamp(val / 3, -1, 1);
    var baseAlpha = p.iaifi ? 1.0 : 0.5;

    if (t >= 0) {
      return [1.0, 1.0 - t * 0.7, 1.0 - t * 0.85, baseAlpha];
    } else {
      var s = -t;
      return [1.0 - s * 0.87, 1.0 - s * 0.42, 1.0, baseAlpha];
    }
  }

  function getPointColor(p) {
    if (colorMode === "category")   return getCategoryColor(p);
    if (colorMode === "enrichment") return getEnrichmentColor(p);
    return getThemeColor(p);
  }

  function getPointSize(p) {
    if (citationSizing && p.iaifi && p.cit != null) {
      return clamp(3 + Math.log2(p.cit + 1), 3, 12);
    }
    return p.iaifi ? SIZE_IAIFI : SIZE_BG;
  }

  // ── Filter logic ───────────────────────────────────────────
  function passesFilters(p) {
    // Theme filter: non-IAIFI (background) always pass
    if (activeTheme !== "all" && p.iaifi && p.theme !== activeTheme) {
      return false;
    }
    // Year filter
    if (p.yr != null && (p.yr < yearMin || p.yr > yearMax)) {
      return false;
    }
    // Search filter
    if (searchQuery) {
      var q = searchQuery.toLowerCase();
      var title   = (p.t || "").toLowerCase();
      var authors = (p.a || "").toLowerCase();
      if (title.indexOf(q) === -1 && authors.indexOf(q) === -1) {
        return false;
      }
    }
    return true;
  }

  function getPointOpacity(p, baseColor) {
    if (!passesFilters(p)) return 0.1;
    return baseColor[3];
  }

  // ── Draw data assembly ─────────────────────────────────────
  function buildDrawArrays() {
    var n = allPapers.length;
    var xs     = new Array(n);
    var ys     = new Array(n);
    var colors = new Array(n);
    var sizes  = new Array(n);

    for (var i = 0; i < n; i++) {
      var p = allPapers[i];
      var c = getPointColor(p);
      var o = getPointOpacity(p, c);

      xs[i]     = p.x;
      ys[i]     = p.y;
      colors[i] = [c[0], c[1], c[2], o];
      sizes[i]  = getPointSize(p);
    }

    return { x: xs, y: ys, color: colors, size: sizes };
  }

  // ── Scatterplot init & draw ────────────────────────────────
  function initScatterplot() {
    var canvas = document.createElement("canvas");
    elContainer.appendChild(canvas);

    scatterplot = createScatterplot({
      canvas: canvas,
      width: elContainer.clientWidth,
      height: elContainer.clientHeight,
      lassoOnLongPress: false,
      // H.2: Enlarge hovered points to 8px
      pointSizeMouseDetection: 8,
      // H.2: Selected points get a bright outline ring
      pointOutlineWidth: 3,
    });

    // Initial draw
    var data = buildDrawArrays();
    scatterplot.draw(data);

    // Resize
    window.addEventListener("resize", function () {
      scatterplot.set({
        width: elContainer.clientWidth,
        height: elContainer.clientHeight,
      });
      updateClusterLabelPositions();
    });

    // Hover -> tooltip + enlarge point to 8px (PLAN H.2)
    var hoveredIdx = null;
    scatterplot.subscribe("pointOver", function (idx) {
      if (idx == null || idx < 0 || idx >= allPapers.length) return;
      hoveredIdx = idx;
      showTooltip(idx);
      // Enlarge the hovered point
      scatterplot.set({ pointSizeSelected: 8 });
      scatterplot.hover(idx);
    });
    scatterplot.subscribe("pointOut", function () {
      hoveredIdx = null;
      hideTooltip();
      scatterplot.hover(null);
    });

    // Click -> side panel + bright ring selection (PLAN H.2)
    scatterplot.subscribe("select", function (selection) {
      if (!selection || selection.length === 0) return;
      var idx = selection[0];
      if (idx == null || idx < 0 || idx >= allPapers.length) return;
      openPanel(allPapers[idx].id);
    });

    // View change -> reposition cluster labels
    scatterplot.subscribe("view", function () {
      updateClusterLabelPositions();
    });
  }

  function applyFilters() {
    if (!scatterplot) return;
    var data = buildDrawArrays();
    scatterplot.draw(data);
    updateClusterLabelPositions();
  }

  // ── Tooltip ────────────────────────────────────────────────
  function showTooltip(idx) {
    var p = allPapers[idx];
    if (!p) return;

    var cat = (p.cat && p.cat.length > 0) ? p.cat[0] : "";
    elTooltip.innerHTML =
      '<div class="tooltip-title">' + escapeHtml(p.t || "Untitled") + '</div>' +
      '<div class="tooltip-meta">' + escapeHtml(p.yr + " \u00B7 " + cat) + '</div>';

    elTooltip.classList.remove("hidden");
    document.addEventListener("mousemove", positionTooltip);
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
    document.removeEventListener("mousemove", positionTooltip);
  }

  // ── Side Panel ─────────────────────────────────────────────
  function openPanel(paperId) {
    var p = paperIndex[paperId];
    if (!p) return;

    panelPaperId = paperId;
    panelNeighborPage = 0;

    elPanelTitle.textContent   = p.t || "Untitled";
    elPanelAuthors.textContent = p.a || "";

    // Meta tags
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

    // Abstract (from abstracts map, or inline field)
    var abs = abstracts[paperId] || p.abstract || "";
    elPanelAbstract.textContent = abs || "(Abstract not available)";

    // arXiv link
    elPanelArxiv.href = "https://arxiv.org/abs/" + paperId;

    // Neighbors
    renderNeighbors();

    // Slide in
    elPanel.classList.remove("hidden");
    void elPanel.offsetHeight; // force reflow for CSS transition
    elPanel.classList.add("visible");
  }

  function closePanel() {
    elPanel.classList.remove("visible");
    setTimeout(function () {
      elPanel.classList.add("hidden");
    }, 300);
    panelPaperId = null;
  }

  function renderNeighbors() {
    var p = paperIndex[panelPaperId];
    if (!p || !p.nn || p.nn.length === 0) {
      elPanelNeighbors.innerHTML = '<li>No neighbors available</li>';
      elShowMore.classList.add("hidden");
      return;
    }

    var end = Math.min((panelNeighborPage + 1) * 5, p.nn.length);
    var visible = p.nn.slice(0, end);

    elPanelNeighbors.innerHTML = "";
    visible.forEach(function (nid) {
      var np = paperIndex[nid];
      var li = document.createElement("li");

      if (np) {
        var cat = (np.cat && np.cat.length > 0) ? np.cat[0] : "";
        li.innerHTML =
          '<div class="neighbor-title">' + escapeHtml(np.t || nid) + '</div>' +
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

    if (end < p.nn.length) {
      elShowMore.classList.remove("hidden");
    } else {
      elShowMore.classList.add("hidden");
    }
  }

  function highlightPoint(paperId) {
    var idx = paperIdxMap[paperId];
    if (idx != null && scatterplot) {
      scatterplot.select([idx]);
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

  // ── Cluster Labels ─────────────────────────────────────────
  function initClusterLabels() {
    clusterLabelEls.forEach(function (el) { el.remove(); });
    clusterLabelEls = [];

    if (!clusters || clusters.length === 0) return;

    clusters.forEach(function (cl) {
      if (cl.cx == null || cl.cy == null) return;
      var label = cl.label || cl.auto_label || "";
      if (!label) return;

      var el = document.createElement("div");
      el.className = "cluster-label";
      el.textContent = label;
      el.dataset.cx = cl.cx;
      el.dataset.cy = cl.cy;
      elContainer.appendChild(el);
      clusterLabelEls.push(el);
    });

    updateClusterLabelPositions();
  }

  function updateClusterLabelPositions() {
    if (!scatterplot || clusterLabelEls.length === 0) return;

    var view;
    try {
      view = scatterplot.get("view");
    } catch (_) {
      return;
    }

    // view is [xStart, yStart, xEnd, yEnd]
    if (!Array.isArray(view) || view.length < 4) return;

    var xRange = Math.abs(view[2] - view[0]);
    var zoomFactor = xRange > 0 ? 1 / xRange : 1;

    var cw = elContainer.clientWidth;
    var ch = elContainer.clientHeight;

    clusterLabelEls.forEach(function (el) {
      var cx = parseFloat(el.dataset.cx);
      var cy = parseFloat(el.dataset.cy);

      // Map data coords to screen coords through the view
      var sx = ((cx - view[0]) / (view[2] - view[0])) * cw;
      var sy = (1 - (cy - view[1]) / (view[3] - view[1])) * ch;

      el.style.left = sx + "px";
      el.style.top  = sy + "px";

      // Show at medium zoom, hide at extremes
      if (zoomFactor < 0.3 || zoomFactor > 30) {
        el.style.opacity = "0";
      } else {
        el.style.opacity = "1";
      }
    });
  }

  // ── Occlusion Diagnostics (H.11) ──────────────────────────
  function runOcclusionDiagnostics() {
    if (!allPapers || allPapers.length === 0) return;

    try {
      var cw = elContainer.clientWidth;
      var ch = elContainer.clientHeight;
      var screenArea = cw * ch;
      if (screenArea === 0) return;

      // Data bounds
      var xMin = Infinity, xMax = -Infinity;
      var yMin = Infinity, yMax = -Infinity;
      allPapers.forEach(function (p) {
        if (p.x < xMin) xMin = p.x;
        if (p.x > xMax) xMax = p.x;
        if (p.y < yMin) yMin = p.y;
        if (p.y > yMax) yMax = p.y;
      });

      var xRange = (xMax - xMin) || 1;
      var yRange = (yMax - yMin) || 1;

      // Screen-space fill ratio
      var totalArea = 0;
      var screenXs = new Float32Array(allPapers.length);
      var screenYs = new Float32Array(allPapers.length);

      allPapers.forEach(function (p, i) {
        var r = p.iaifi ? SIZE_IAIFI : SIZE_BG;
        totalArea += Math.PI * r * r;
        screenXs[i] = ((p.x - xMin) / xRange) * cw;
        screenYs[i] = ((p.y - yMin) / yRange) * ch;
      });

      var fillRatio = totalArea / screenArea;

      // Median NN screen distance (sampled for performance)
      var sampleStep = Math.max(1, Math.floor(allPapers.length / 500));
      var nnDists = [];

      for (var i = 0; i < allPapers.length; i += sampleStep) {
        var minD = Infinity;
        for (var j = 0; j < allPapers.length; j++) {
          if (i === j) continue;
          var dx = screenXs[i] - screenXs[j];
          var dy = screenYs[i] - screenYs[j];
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < minD) minD = d;
        }
        if (minD < Infinity) nnDists.push(minD);
      }

      nnDists.sort(function (a, b) { return a - b; });
      var medianNN = nnDists.length > 0 ? nnDists[Math.floor(nnDists.length / 2)] : 0;

      console.log("[Occlusion Diagnostics]");
      console.log("  Screen: " + cw + "x" + ch + " (" + screenArea + " px^2)");
      console.log("  Point area: " + Math.round(totalArea) + " px^2");
      console.log("  Fill ratio: " + (fillRatio * 100).toFixed(2) + "%");
      console.log("  Median NN screen dist: " + medianNN.toFixed(1) + " px");
      console.log("  Points: " + allPapers.length);

      if (fillRatio > 0.3) {
        console.warn("[Occlusion] Fill ratio " + (fillRatio * 100).toFixed(1) +
          "% > 30%. Consider multiscale rendering (v2.1b).");
      }
      if (medianNN < 3) {
        console.warn("[Occlusion] Median NN dist " + medianNN.toFixed(1) +
          "px < 3px. Heavy overlap likely. Consider multiscale rendering (v2.1b).");
      }
    } catch (err) {
      console.warn("[Occlusion Diagnostics] Error:", err.message);
    }
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

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && elPanel.classList.contains("visible")) {
        closePanel();
      }
    });
  }

  // ── Main ───────────────────────────────────────────────────
  async function main() {
    cacheDom();
    bindEvents();

    // Load primary data
    var data;
    try {
      var resp = await fetch("data/papers.json");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      data = await resp.json();
    } catch (err) {
      elContainer.innerHTML =
        '<p style="padding:40px;text-align:center;color:#c00;">' +
        'Error loading data. Make sure <code>data/papers.json</code> exists.<br>' +
        escapeHtml(err.message) + '</p>';
      return;
    }

    meta      = data.meta || {};
    clusters  = data.clusters || [];
    abstracts = data.abstracts || {};
    allPapers = data.papers || [];

    // Build cluster lookup
    clusters.forEach(function (c) { clusterMap[c.id] = c; });

    // Try loading split abstracts file if none inline
    if (Object.keys(abstracts).length === 0) {
      try {
        var absResp = await fetch("data/abstracts.json");
        if (absResp.ok) {
          var absData = await absResp.json();
          abstracts = absData.abstracts || absData;
        }
      } catch (_) { /* graceful degradation */ }
    }

    // Build paper indices
    allPapers.forEach(function (p, i) {
      paperIndex[p.id]  = p;
      paperIdxMap[p.id] = i;
    });

    // Citation data detection
    hasCitations = allPapers.some(function (p) {
      return p.cit != null && p.cit > 0;
    });
    if (!hasCitations && elCitWrap) {
      elCitWrap.classList.add("hidden");
    }

    // Year bounds from data
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

    // Display data version
    if (meta.version) {
      var parts = [];
      if (meta.generated) parts.push(meta.generated.slice(0, 10));
      if (meta.embedding_model) parts.push(meta.embedding_model);
      if (meta.n_total) parts.push(meta.n_total.toLocaleString() + " papers");
      elDataVersion.textContent = "Data: " + parts.join(" | ");
    }

    // Initialize UI components
    initSliders();
    initScatterplot();
    initClusterLabels();

    // Run diagnostics after a short delay to let layout settle
    requestAnimationFrame(function () {
      runOcclusionDiagnostics();
    });
  }

  document.addEventListener("DOMContentLoaded", main);

})();
