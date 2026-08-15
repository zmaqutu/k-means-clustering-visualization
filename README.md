# K-Means 3D Lab

An interactive Three.js visualization of the K-means clustering algorithm. The lab turns the original Python implementation in `machine-learning-k-means-clustering` into a step-by-step spatial experience, with the visual language of `visualize-word-embeddings`.

## What you can explore

- Alternate between point assignment and centroid update steps.
- Run automatically until convergence or advance one operation at a time.
- Compare random, farthest-point, and K-means++ initialization.
- Change K from 2–6 and generate up to 600 observations.
- Explore compact and unequal Gaussian groups, anisotropic ribbons, interlocking moons, overlapping currents, concentric shells, uniform noise, or the original eight Python points.
- Toggle the continuous Voronoi decision field, point-to-centroid distance lines, and camera orbiting.
- Clear centroid seeds and place them manually on observations, or add genuinely random seeds one at a time.
- Watch inertia, moved points, centroid shift, cluster sizes, and centroid trails update live.

## Local development

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Production build

```bash
npm run build
```

## GitHub Pages

Every push to `main` runs `.github/workflows/deploy-pages.yml`. The workflow builds a dedicated static Vite entry and publishes `dist-pages` through GitHub's official Pages actions. For the first deployment, set the repository's **Settings → Pages → Source** to **GitHub Actions**.

To verify the static build locally:

```bash
npm run build:pages
```

## Controls

- Drag: orbit the 3D scene
- Scroll: zoom
- Space: run one algorithm step
- A: toggle automatic playback
- R: reset the current sample

Built with React, Three.js, React Three Fiber, Drei, and the Sites-compatible Vite/Vinext runtime.
