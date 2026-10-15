// Render an animation page to an mp4, one deterministic frame at a time.
//
//   npm install puppeteer-core
//   node tools/animations/render.js tools/animations/teaser.html static/videos/gtp-teaser.mp4
//
// Serves the repository root over HTTP (the pages load the site's videos by relative path) and
// drives the local Google Chrome. Options: --fps 30, --from 0 --to <seconds> for a partial render,
// --frames dir/ to write PNGs instead of a video, --stills 2,6.5 dir/ for single frames.
//
// Slides: one PNG per scene, at the times the page lists in window.SLIDES:
//   node tools/animations/render.js tools/animations/teaser.html tools/animations/slides/teaser --slides
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const puppeteer = require('puppeteer-core');

const ROOT = path.resolve(__dirname, '..', '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.mp4': 'video/mp4', '.png': 'image/png', '.svg': 'image/svg+xml' };

function arg(name, fallback) {
  const i = process.argv.indexOf('--' + name);
  return i > 0 ? process.argv[i + 1] : fallback;
}

// Static server with Range support, which Chrome needs to seek in videos.
function serve() {
  const server = http.createServer((req, res) => {
    const file = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      return res.end();
    }
    const size = fs.statSync(file).size;
    const type = TYPES[path.extname(file)] || 'application/octet-stream';
    const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
    if (range) {
      const start = range[1] ? Number(range[1]) : 0;
      const end = range[2] ? Number(range[2]) : size - 1;
      res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 });
      fs.createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': size });
      fs.createReadStream(file).pipe(res);
    }
  });
  return new Promise((resolve) => server.listen(0, () => resolve(server)));
}

(async () => {
  const [page_, out] = process.argv.slice(2);
  const fps = Number(arg('fps', 30));
  const framesDir = arg('frames', null);
  const server = await serve();
  // The page may carry a query string (e.g. method.html?H=3), which is passed through to it.
  const [pageFile, query] = page_.split('?');
  const url = `http://localhost:${server.address().port}/${path.relative(ROOT, path.resolve(pageFile))}${query ? '?' + query : ''}`;

  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--autoplay-policy=no-user-gesture-required', '--hide-scrollbars'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  page.on('console', (m) => console.log('[page]', m.text()));
  page.on('pageerror', (e) => console.error('[page error]', e.message));
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.evaluate(() => window.pageReady);
  const size = await page.evaluate(() => window.SIZE || [1920, 1080]);
  await page.setViewport({ width: size[0], height: size[1], deviceScaleFactor: 1 });
  const duration = await page.evaluate(() => window.DURATION);
  const from = Number(arg('from', 0));
  const to = Number(arg('to', duration));
  const n = Math.round((to - from) * fps);

  // --stills 2,6.5,10 out_dir: save single frames for checking a layout.
  // --slides out_dir: save one frame per slide listed in window.SLIDES ([name, seconds] pairs).
  const stills = arg('stills', null);
  const slides = process.argv.includes('--slides') ? await page.evaluate(() => window.SLIDES || null) : null;
  if (process.argv.includes('--slides') && !slides) throw new Error('the page defines no window.SLIDES');
  if (stills || slides) {
    fs.mkdirSync(out, { recursive: true });
    const shots = slides
      ? slides.map(([name, t], i) => ({ t, file: `${String(i + 1).padStart(2, '0')}-${name}.png` }))
      : stills.split(',').map(Number).map((t) => ({ t, file: `t${t.toFixed(2)}.png` }));
    for (const { t, file } of shots) {
      await page.evaluate((t) => window.renderAt(t), t);
      await page.screenshot({ path: path.join(out, file) });
      if (slides) console.log(`${file}  (t = ${t.toFixed(2)}s)`);
    }
    await browser.close();
    server.close();
    return;
  }

  let ffmpeg = null;
  if (framesDir) fs.mkdirSync(framesDir, { recursive: true });
  else {
    ffmpeg = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  }

  for (let i = 0; i < n; i++) {
    const t = from + i / fps;
    await page.evaluate((t) => window.renderAt(t), t);
    const png = await page.screenshot({ type: 'png', optimizeForSpeed: true });
    if (framesDir) fs.writeFileSync(path.join(framesDir, `f${String(i).padStart(5, '0')}.png`), png);
    else if (!ffmpeg.stdin.write(png)) await new Promise((r) => ffmpeg.stdin.once('drain', r));
    if (i % fps === 0) process.stdout.write(`\r${t.toFixed(1)}s / ${to}s`);
  }
  process.stdout.write('\n');
  await browser.close();
  server.close();
  if (ffmpeg) {
    ffmpeg.stdin.end();
    await new Promise((r) => ffmpeg.on('close', r));
  }
})();
