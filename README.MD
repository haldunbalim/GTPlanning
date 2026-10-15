# Generative Trajectory Planning Project Page

This repository contains the project website for:

```
Model-Based Online Decision Making via Generative Trajectory Planning
```

Citation:

```bibtex
@inproceedings{balim2026modelbasedonline,
  title     = {Model-Based Online Decision Making via Generative Trajectory Planning},
  author    = {Balim, Haldun and Du, Yilun and Li, Na},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Website License

The site structure is adapted from the Nerfies project page template, which is distributed under the Creative Commons Attribution-ShareAlike 4.0 International License.

## Animations

The teaser (`static/videos/gtp-teaser.mp4`) and method (`static/videos/gtp-method.mp4`) videos are rendered from the HTML pages in `tools/animations/`, which read the benchmark data and rollout videos from `static/`. Open a page with `?play` to preview it, then render it frame by frame with headless Chrome:

```bash
npm install --no-save --prefix tools/animations puppeteer-core
export NODE_PATH=tools/animations/node_modules
node tools/animations/render.js tools/animations/teaser.html static/videos/gtp-teaser.mp4
node tools/animations/render.js tools/animations/method.html static/videos/gtp-method.mp4
```

To review the slides without rendering a video, export one image per scene (the times are listed in each page's `window.SLIDES`) to `tools/animations/slides/`, which git ignores:

```bash
node tools/animations/render.js tools/animations/teaser.html tools/animations/slides/teaser --slides
node tools/animations/render.js tools/animations/method.html tools/animations/slides/method --slides
```
