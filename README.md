# K-Means 3D Lab

An interactive Three.js visualization of the K-means clustering algorithm. The lab turns the original Python implementation in `machine-learning-k-means-clustering` into a step-by-step spatial experience, with the visual language of `visualize-word-embeddings`.

## What you can explore

- Alternate between point assignment and centroid update steps.
- Run automatically until convergence or advance one operation at a time.
- Compare random, farthest-point, and K-means++ initialization.
- Change K from 2–6 and generate up to 300 observations.
- Try compact Gaussian groups, overlapping elongated clouds, concentric shells, or the original eight Python points.
- Toggle centroid fields, point-to-centroid distance lines, and camera orbiting.
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

## Controls

- Drag: orbit the 3D scene
- Scroll: zoom
- Space: run one algorithm step
- A: toggle automatic playback
- R: reset the current sample

Built with React, Three.js, React Three Fiber, Drei, and the Sites-compatible Vite/Vinext runtime.
