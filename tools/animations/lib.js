// Shared helpers for the frame-accurate animations. Each page defines window.DURATION and a
// draw(t) function; renderAt(t) draws a frame and waits until every video has seeked to it,
// so render.js can capture deterministic frames. Open a page with ?play to preview it live,
// or ?t=12.5 to inspect a single frame.
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';

  var A = window.A = {};

  A.COLORS = {
    navy: '#102033',
    ink: '#263548',
    muted: '#5b6777',
    line: '#d7ddd9',
    paper: '#faf9f4',
    sac: '#76B7B2',
    dpmd: '#D55E00',
    tdmpc2: '#009E73',
    gtp: '#5DBDF3',
    gtpDeep: '#1f8fd0',
    red: '#a3403f'
  };

  A.clamp = function (x, lo, hi) { return Math.max(lo, Math.min(hi, x)); };
  A.lerp = function (a, b, u) { return a + (b - a) * u; };
  // Progress of t through [a, b], clamped to [0, 1].
  A.prog = function (t, a, b) { return A.clamp((t - a) / (b - a), 0, 1); };
  A.easeOut = function (u) { return 1 - Math.pow(1 - u, 3); };
  A.easeInOut = function (u) { return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; };
  A.easeBack = function (u) { var c = 1.7; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };

  // Opacity of something shown in [a, b] with fades of length f at both ends.
  A.window = function (t, a, b, f) {
    f = f || 0.45;
    return Math.min(A.prog(t, a, a + f), 1 - A.prog(t, b - f, b));
  };

  // Seeded RNG so every render draws the same trajectories.
  A.rng = function (seed) {
    var s = seed >>> 0;
    var next = function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var r = Math.imul(s ^ (s >>> 15), 1 | s);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
    next.gauss = function () {
      var u = 1 - next(), v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return next;
  };

  A.svg = function (tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    A.attrs(node, attrs);
    if (parent) parent.appendChild(node);
    return node;
  };

  A.html = function (tag, attrs, parent) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(node);
    return node;
  };

  A.attrs = function (node, attrs) {
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    return node;
  };

  // Show an element with an opacity and an upward slide as it fades in.
  A.show = function (el, o, dy, scale) {
    el.style.opacity = o;
    el.style.visibility = o <= 0.001 ? 'hidden' : 'visible';
    var tr = 'translateY(' + ((1 - Math.min(1, o)) * (dy || 0)) + 'px)';
    if (scale !== undefined) tr += ' scale(' + scale + ')';
    el.style.transform = tr;
  };

  // Smooth path through points (Catmull-Rom as cubic Beziers). Returns one segment string per
  // gap so callers can style the segments separately.
  A.splineSegments = function (pts) {
    var segs = [];
    for (var i = 0; i < pts.length - 1; i++) {
      var p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
      var c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      var c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      segs.push('M' + p1[0].toFixed(1) + ',' + p1[1].toFixed(1) + 'C' + c1[0].toFixed(1) + ',' + c1[1].toFixed(1) +
        ' ' + c2[0].toFixed(1) + ',' + c2[1].toFixed(1) + ' ' + p2[0].toFixed(1) + ',' + p2[1].toFixed(1));
    }
    return segs;
  };

  A.spline = function (pts) {
    return A.splineSegments(pts).map(function (s, i) { return i ? s.replace(/^M[^C]+/, '') : s; }).join('');
  };

  // Part of a polyline/spline up to fraction u of its points (for drawing a curve in over time).
  A.partial = function (pts, u) {
    var n = (pts.length - 1) * A.clamp(u, 0, 1);
    var k = Math.floor(n), f = n - k;
    var out = pts.slice(0, k + 1);
    if (k < pts.length - 1 && f > 0) {
      out.push([A.lerp(pts[k][0], pts[k + 1][0], f), A.lerp(pts[k][1], pts[k + 1][1], f)]);
    }
    return out;
  };

  // ---- Videos: seek every registered clip to its local time and wait for the frame. ----
  var videos = [];

  // `start` is when the clip begins on the page timeline. Clips loop, unless `once` is set: then the
  // clip plays through a single time and holds its last frame.
  A.video = function (src, start, parent, cls, once) {
    var v = A.html('video', { class: cls || '', muted: '', playsinline: '', preload: 'auto' }, parent);
    v.muted = true;
    v.src = src;
    videos.push({ el: v, start: start || 0, once: Boolean(once) });
    return v;
  };

  A.videosReady = function () {
    return Promise.all(videos.map(function (v) {
      return new Promise(function (resolve) {
        if (v.el.readyState >= 2) return resolve();
        v.el.addEventListener('loadeddata', resolve, { once: true });
        v.el.addEventListener('error', resolve, { once: true });
      });
    }));
  };

  function seekVideo(v, t) {
    var el = v.el;
    if (!el.duration || el.style.visibility === 'hidden' || isHidden(el)) return null;
    var end = Math.max(0.1, el.duration - 0.05), elapsed = Math.max(0, t - v.start);
    var local = v.once ? Math.min(elapsed, end) : elapsed % end;
    if (Math.abs(el.currentTime - local) < 1e-3) return null;
    return new Promise(function (resolve) {
      var done = function () { el.removeEventListener('seeked', done); resolve(); };
      el.addEventListener('seeked', done);
      el.currentTime = local;
      setTimeout(done, 1500);
    });
  }

  function isHidden(el) {
    for (var n = el; n && n !== document.body; n = n.parentElement) {
      if (n.style && (n.style.visibility === 'hidden' || n.style.display === 'none')) return true;
    }
    return false;
  }

  // Extra things to wait for before the first frame (e.g. an embedded animation page).
  var pending = [];
  A.waitFor = function (promise) { pending.push(promise); };

  A.renderAt = function (t) {
    var drawn = window.draw(t);  // may return a promise (e.g. an embedded page's frame)
    // Forcing layout makes the browser request any font subset the new text needs (web fonts
    // load glyphs such as ✕ lazily); waiting for it avoids frames where that text is invisible.
    void document.body.offsetHeight;
    var waits = videos.map(function (v) { return seekVideo(v, t); }).filter(Boolean);
    waits.push(document.fonts.ready);
    if (drawn && drawn.then) waits.push(drawn);
    return Promise.all(waits);
  };
  window.renderAt = A.renderAt;

  A.start = function () {
    var params = new URLSearchParams(location.search);
    var ready = Promise.all([A.videosReady(), document.fonts.ready].concat(pending));
    window.pageReady = ready.then(function () { return A.renderAt(0); });
    if (params.has('t')) {
      window.pageReady.then(function () { return A.renderAt(parseFloat(params.get('t'))); });
    } else if (params.has('play')) {
      window.pageReady.then(function () {
        var t0 = performance.now();
        var busy = false;
        (function tick() {
          if (!busy) {
            busy = true;
            A.renderAt(((performance.now() - t0) / 1000) % window.DURATION).then(function () { busy = false; });
          }
          requestAnimationFrame(tick);
        })();
      });
    }
  };
})();
