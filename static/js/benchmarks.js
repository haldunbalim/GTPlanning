// Interactive benchmark explorer: environment tabs, the paper's learning curves with a checkpoint
// slider under them, and rollout videos per method beside the plot. Data comes from benchmarks-data.js.
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var PLOT = { width: 560, height: 380, left: 64, right: 16, top: 14, bottom: 46 };

  function el(tag, attrs, parent) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === 'text') node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function svgEl(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === 'text') node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function formatSteps(step) {
    return step >= 1000000 ? (step / 1000000) + 'M' : (step / 1000) + 'k';
  }

  function formatReturn(value, scale) {
    if (scale >= 100) return Math.round(value).toLocaleString('en-US');
    return value.toFixed(1);
  }

  function formatTick(value, scale) {
    var rounded = Math.round(value * 1e6) / 1e6 || 0;  // `|| 0` turns -0 into 0
    return Number.isInteger(rounded) ? rounded.toLocaleString('en-US') : formatReturn(rounded, scale);
  }

  // Round tick spacing to 1, 2 or 5 times a power of ten.
  function niceTicks(min, max, count) {
    var raw = (max - min) / count;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 5, 10].map(function (m) { return m * mag; })
      .find(function (s) { return s >= raw; });
    var ticks = [];
    for (var t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(t);
    return ticks;
  }

  function Explorer(root, data, group) {
    this.root = root;
    this.data = data;
    this.envs = group.envs;
    this.metric = group.metric;  // {label, unit, range?}
    // What each tile / curve shows (methods, horizons, ablations); a tab may override it.
    this.groupSeries = group.series || data.algos;
    this.envIndex = 0;
    this.steps = group.steps || data.steps;  // slider checkpoints
    this.curveSteps = group.curve_steps || data.curve_steps;  // x positions of the curve points
    this.xMin = group.x_min || 0;  // where the plot's x-axis starts
    this.stepIndex = this.steps.length - 1;
    this.build();
    this.selectEnv(0);
  }

  Explorer.prototype.build = function () {
    var self = this;
    var id = this.root.getAttribute('data-group');

    // The tabs sit next to the section title when the page provides a slot for them;
    // a section with a single environment shows none.
    var tabs = document.querySelector('.env-tabs[data-group="' + id + '"]') ||
      el('div', { class: 'env-tabs' }, this.root);
    tabs.setAttribute('role', 'group');
    tabs.setAttribute('aria-label', 'Environment');
    tabs.style.display = this.envs.length > 1 ? '' : 'none';
    this.tabs = this.envs.map(function (env, i) {
      var button = el('button', { type: 'button', class: 'env-tab', text: env.label, 'aria-pressed': 'false' }, tabs);
      button.addEventListener('click', function () { self.selectEnv(i); });
      return button;
    });

    this.description = el('p', { class: 'benchmark-description' }, this.root);
    // The plot and its slider sit on the left, the rollout videos on the right.
    this.body = el('div', { class: 'explorer-body' }, this.root);
    var plotColumn = this.plotColumn = el('div', { class: 'explorer-plot' }, this.body);

    var figure = el('figure', { class: 'paper-figure benchmark-plot' }, plotColumn);
    this.svg = svgEl('svg', {
      viewBox: '0 0 ' + PLOT.width + ' ' + PLOT.height, role: 'img',
      'aria-label': 'Evaluation return over training for each method'
    }, figure);
    this.legend = el('div', { class: 'plot-legend' }, figure);

    var control = el('div', { class: 'step-control' }, plotColumn);
    var label = el('label', { for: id + '-step', class: 'step-label' }, control);
    label.appendChild(document.createTextNode('Checkpoint: '));
    this.stepOutput = el('strong', {}, label);
    label.appendChild(document.createTextNode(' environment steps'));
    this.slider = el('input', {
      type: 'range', id: id + '-step', class: 'step-slider', min: '0',
      max: String(this.steps.length - 1), step: '1', value: String(this.stepIndex)
    }, control);
    var ticks = el('div', { class: 'step-ticks', 'aria-hidden': 'true' }, control);
    this.steps.forEach(function (step) { el('span', { text: formatSteps(step) }, ticks); });
    this.slider.addEventListener('input', function () { self.selectStep(Number(self.slider.value)); });

    this.grid = el('div', { class: 'rollout-grid benchmark-videos' }, this.body);

    // Keep a 2x2 grid of videos exactly as tall as the plot and slider beside it.
    if (window.ResizeObserver) new ResizeObserver(function () { self.fitVideos(); }).observe(plotColumn);
  };

  // A 2x2 grid of square videos is as tall as it is wide plus two caption rows, so narrow it until
  // its height matches the plot column. Below the two-column breakpoint the grid fills its width.
  Explorer.prototype.fitVideos = function () {
    var grid = this.grid;
    if (!this.tiles || this.tiles.length !== 4) return;
    grid.style.width = '';
    if (getComputedStyle(this.body).gridTemplateColumns.split(' ').length < 2) return;
    var gap = parseFloat(getComputedStyle(grid).rowGap) || 0;
    var caption = this.tiles[0].video.parentNode.querySelector('figcaption').offsetHeight;
    var border = this.tiles[0].video.parentNode.offsetHeight - this.tiles[0].video.parentNode.clientHeight;
    var side = (this.plotColumn.offsetHeight - gap) / 2 - caption - border;
    grid.style.width = Math.min(grid.offsetWidth, 2 * (side + border) + gap) + 'px';  // never wider than its column
  };

  // (Re)build the video tiles and plot legend for a set of series.
  Explorer.prototype.setSeries = function (series) {
    var grid = this.grid, legend = this.legend;
    this.series = series;
    while (grid.firstChild) grid.removeChild(grid.firstChild);
    while (legend.firstChild) legend.removeChild(legend.firstChild);
    grid.className = 'rollout-grid benchmark-videos has-' + series.length;
    this.body.className = 'explorer-body has-' + series.length;
    this.tiles = series.map(function (algo) {
      var tile = el('figure', { class: 'paper-figure rollout-tile benchmark-tile' + (algo.key === 'gtp' ? ' is-ours' : '') }, grid);
      var video = el('video', { class: 'rollout-video', muted: '', loop: '', playsinline: '', preload: 'auto' }, tile);
      video.muted = true;
      var placeholder = el('div', { class: 'video-placeholder', text: 'Video coming soon' }, tile);
      var caption = el('figcaption', {}, tile);
      var name = el('span', { class: 'tile-name' }, caption);
      el('span', { class: 'tile-swatch', style: 'background:' + algo.color }, name);
      name.appendChild(document.createTextNode(algo.label));
      var value = el('span', { class: 'tile-value' }, caption);
      return { algo: algo, video: video, placeholder: placeholder, value: value };
    });
    series.forEach(function (algo) {
      var item = el('span', { class: 'plot-legend-item' }, legend);
      el('span', { class: 'tile-swatch', style: 'background:' + algo.color }, item);
      item.appendChild(document.createTextNode(algo.label));
    });
  };

  Explorer.prototype.env = function () {
    return this.envs[this.envIndex];
  };

  Explorer.prototype.formatValue = function (mean, se) {
    var unit = this.metric.unit;
    if (se === null || se === undefined) {  // a single run: no standard error
      return (unit === '%' ? Math.round(mean) : formatReturn(mean, this.scale)) + unit;
    }
    if (unit === '%') return Math.round(mean) + ' ± ' + Math.round(se) + unit;
    return formatReturn(mean, this.scale) + ' ± ' + formatReturn(se, this.scale);
  };

  // Value of one method at a checkpoint: from its curve, or its final value at the last checkpoint.
  Explorer.prototype.valueAt = function (curve, curveIndex) {
    if (curve.mean) return { mean: curve.mean[curveIndex], se: curve.se ? curve.se[curveIndex] : null };
    if (curveIndex === this.curveSteps.length - 1) return curve.final;
    return null;
  };

  Explorer.prototype.selectEnv = function (i) {
    var env = this.envs[i];
    var series = env.series || this.groupSeries;
    this.envIndex = i;
    this.tabs.forEach(function (tab, j) {
      tab.classList.toggle('is-active', j === i);
      tab.setAttribute('aria-pressed', j === i ? 'true' : 'false');
    });
    if (series !== this.series) {
      this.setSeries(series);
      this.fitVideos();
    }
    this.description.textContent = env.description || '';
    this.description.style.display = env.description ? '' : 'none';
    this.drawPlot();
    this.selectStep(this.stepIndex);
  };

  Explorer.prototype.selectStep = function (i) {
    var self = this;
    var env = this.env();
    var step = this.steps[i];
    var curveIndex = this.curveSteps.indexOf(step);
    this.stepIndex = i;
    this.slider.value = String(i);
    this.slider.setAttribute('aria-valuetext', formatSteps(step) + ' environment steps');
    this.stepOutput.textContent = formatSteps(step);

    this.tiles.forEach(function (tile) {
      var value = self.valueAt(env.curves[tile.algo.key], curveIndex);
      tile.value.textContent = value
        ? self.metric.label + ' ' + self.formatValue(value.mean, value.se)
        : self.metric.label + ' logged only at the end of training';
    });
    this.playVideos(this.tiles.map(function (tile) { return env.videos[tile.algo.key][i]; }));
    this.moveMarker(step, curveIndex);
  };

  // Swap all videos together and start them on the same frame. A null source shows a placeholder.
  Explorer.prototype.playVideos = function (sources) {
    var token = (this.playToken || 0) + 1;
    var self = this;
    var ready = 0;
    var active = this.tiles.filter(function (tile, j) { return sources[j]; });
    this.playToken = token;
    this.tiles.forEach(function (tile, j) {
      var video = tile.video;
      var hasVideo = Boolean(sources[j]);
      video.pause();
      video.style.display = hasVideo ? '' : 'none';
      tile.placeholder.style.display = hasVideo ? 'none' : '';
      if (!hasVideo) {
        video.removeAttribute('src');
        video.load();
        return;
      }
      // A video that fails to load must not hold back the others.
      video.oncanplay = video.onerror = function () {
        video.oncanplay = video.onerror = null;
        ready += 1;
        if (ready === active.length && self.playToken === token) {
          active.forEach(function (t) {
            t.video.currentTime = 0;
            var p = t.video.play();
            if (p && p.catch) p.catch(function () {});
          });
        }
      };
      video.src = sources[j];
      video.load();
    });
  };

  Explorer.prototype.drawPlot = function () {
    var svg = this.svg;
    var env = this.env();
    var steps = this.curveSteps;
    var algos = this.series;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var lo = Infinity, hi = -Infinity;
    algos.forEach(function (algo) {
      var c = env.curves[algo.key];
      var means = c.mean || [c.final.mean];
      var ses = c.mean ? (c.se || c.mean.map(function () { return 0; })) : [c.final.se];
      means.forEach(function (m, k) {
        lo = Math.min(lo, m - ses[k]);
        hi = Math.max(hi, m + ses[k]);
      });
    });
    if (this.metric.range) {
      lo = this.metric.range[0];
      hi = this.metric.range[1];
    }
    var pad = (hi - lo) * 0.04;
    lo -= pad;
    hi += pad;
    this.scale = Math.max(Math.abs(lo), Math.abs(hi));
    var unit = this.metric.unit;
    var range = this.metric.range;

    var x0 = PLOT.left, x1 = PLOT.width - PLOT.right;
    var y0 = PLOT.height - PLOT.bottom, y1 = PLOT.top;
    var xMax = steps[steps.length - 1];
    var xMin = this.xMin;
    var sx = function (s) { return x0 + (x1 - x0) * (s - xMin) / (xMax - xMin); };
    var sy = function (v) { return y0 + (y1 - y0) * (v - lo) / (hi - lo); };
    this.sx = sx;
    this.sy = sy;

    var scale = this.scale;
    var axes = svgEl('g', { class: 'plot-axes' }, svg);
    niceTicks(range ? range[0] : lo, range ? range[1] : hi, 5).forEach(function (t) {
      svgEl('line', { x1: x0, x2: x1, y1: sy(t), y2: sy(t), class: 'plot-grid' }, axes);
      svgEl('text', { x: x0 - 8, y: sy(t) + 4, 'text-anchor': 'end', class: 'plot-tick', text: formatTick(t, scale) + unit }, axes);
    });
    (xMin ? [] : [0]).concat(this.steps).forEach(function (s) {  // an offset axis is labeled at checkpoints only
      svgEl('line', { x1: sx(s), x2: sx(s), y1: y0, y2: y0 + 5, class: 'plot-axis' }, axes);
      svgEl('text', { x: sx(s), y: y0 + 19, 'text-anchor': 'middle', class: 'plot-tick', text: s ? formatSteps(s) : '0' }, axes);
    });
    svgEl('line', { x1: x0, x2: x1, y1: y0, y2: y0, class: 'plot-axis' }, axes);
    svgEl('line', { x1: x0, x2: x0, y1: y0, y2: y1, class: 'plot-axis' }, axes);
    svgEl('text', { x: (x0 + x1) / 2, y: PLOT.height - 6, 'text-anchor': 'middle', class: 'plot-label', text: 'Environment steps' }, axes);
    svgEl('text', {
      x: 16, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'plot-label plot-ylabel',
      transform: 'rotate(-90 16 ' + (y0 + y1) / 2 + ')', text: this.metric.label
    }, axes);
    svg.setAttribute('aria-label', 'Evaluation ' + this.metric.label.toLowerCase() + ' over training for each method');

    this.marker = svgEl('line', { y1: y0, y2: y1, class: 'plot-marker' }, svg);
    this.dots = algos.map(function (algo) {
      var c = env.curves[algo.key];
      if (!c.mean) {  // final value only: a point with an error bar at the end of training
        var xf = sx(steps[steps.length - 1]);
        svgEl('line', {
          x1: xf, x2: xf, y1: sy(c.final.mean - c.final.se), y2: sy(c.final.mean + c.final.se),
          stroke: algo.color, class: 'plot-errorbar'
        }, svg);
        svgEl('circle', { cx: xf, cy: sy(c.final.mean), r: 4, stroke: algo.color, class: 'plot-final' }, svg);
        return svgEl('circle', { r: 5, fill: algo.color, class: 'plot-dot' }, svg);
      }
      if (c.se) {
        var upper = c.mean.map(function (m, k) { return sx(steps[k]) + ',' + sy(m + c.se[k]); });
        var lower = c.mean.map(function (m, k) { return sx(steps[k]) + ',' + sy(m - c.se[k]); }).reverse();
        svgEl('polygon', { points: upper.concat(lower).join(' '), fill: algo.color, class: 'plot-band' }, svg);
      }
      if (c.mean.length <= 6) {  // few checkpoints: mark each one
        c.mean.forEach(function (m, k) {
          svgEl('circle', { cx: sx(steps[k]), cy: sy(m), r: 3, fill: algo.color, class: 'plot-point' }, svg);
        });
      }
      svgEl('polyline', {
        points: c.mean.map(function (m, k) { return sx(steps[k]) + ',' + sy(m); }).join(' '),
        stroke: algo.color, class: 'plot-line' + (algo.key === 'gtp' ? ' is-ours' : '')
      }, svg);
      return svgEl('circle', { r: 5, fill: algo.color, class: 'plot-dot' }, svg);
    });
  };

  Explorer.prototype.moveMarker = function (step, curveIndex) {
    var env = this.env();
    var sx = this.sx, sy = this.sy;
    this.marker.setAttribute('x1', sx(step));
    this.marker.setAttribute('x2', sx(step));
    this.series.forEach(function (algo, j) {
      var value = this.valueAt(env.curves[algo.key], curveIndex);
      this.dots[j].style.display = value ? '' : 'none';
      if (!value) return;
      this.dots[j].setAttribute('cx', sx(step));
      this.dots[j].setAttribute('cy', sy(value.mean));
    }, this);
  };

  document.addEventListener('DOMContentLoaded', function () {
    var data = window.GTP_BENCHMARKS;
    if (!data) return;
    document.querySelectorAll('.benchmark-explorer').forEach(function (root) {
      var group = data.groups[root.getAttribute('data-group')];
      if (group) new Explorer(root, data, group);
    });
  });
})();
