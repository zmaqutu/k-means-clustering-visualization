<h1 align="center">Clustering 3D Lab</h1>

<p align="center">
  An interactive, step-by-step 3D laboratory for centroid, medoid, density, and probabilistic clustering.
</p>

<div align="center">
  <img alt="Made by Zongo Maqutu" src="https://img.shields.io/badge/made%20by-Zongo%20Maqutu-76e4f7?style=for-the-badge&labelColor=101820" />
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-101820?style=for-the-badge&logo=typescript&logoColor=76e4f7" />
  <img alt="React" src="https://img.shields.io/badge/React%2019-101820?style=for-the-badge&logo=react&logoColor=76e4f7" />
  <img alt="Three.js" src="https://img.shields.io/badge/Three.js-101820?style=for-the-badge&logo=threedotjs&logoColor=ffffff" />
  <img alt="React Three Fiber" src="https://img.shields.io/badge/React%20Three%20Fiber-101820?style=for-the-badge&logo=react&logoColor=f6d65f" />
  <img alt="Vite" src="https://img.shields.io/badge/Vite-101820?style=for-the-badge&logo=vite&logoColor=c4a7ff" />
</div>

<br />

<p align="center">
  <img src="./readmeAssets/clustering-3d-lab-banner.png" width="100%" alt="Clustering 3D Lab cover depicting K-means, K-medoids, DBSCAN, and Gaussian mixture clustering in a luminous three-dimensional coordinate space" />
</p>

<p align="center"><em>Centroids, medoids, density-connected regions, noise, and probabilistic Gaussian components—all in one spatial laboratory.</em></p>

## Live demo

Explore the production version at **[zmaqutu.github.io/k-means-clustering-visualization/](https://zmaqutu.github.io/k-means-clustering-visualization/)**.

Clustering 3D Lab turns four foundational clustering methods into a spatial experiment. Compare their assumptions on the same datasets, run an algorithm automatically, advance it one operation at a time, inspect observations, and deliberately test each method on geometry it handles poorly.

<p align="center">
  <img src="./readmeAssets/kmeans-mode-interface.png" width="100%" alt="Clustering 3D Lab in K-means mode showing a converged double-helix dataset, controls, cluster hulls, and live diagnostics" />
</p>

<p align="center"><sub>K-means mode on the double-helix dataset. The same laboratory also runs K-medoids, DBSCAN, and Gaussian mixture models.</sub></p>

## Highlights

- **Four real clustering engines** — switch between K-means, K-medoids, DBSCAN, and a spherical Gaussian mixture model.
- **Step-by-step playback** — inspect assignments and parameter updates instead of jumping directly to the final partition.
- **Automatic convergence** — play every operation until assignments and centroid positions settle.
- **Three initialization strategies** — compare random, K-means++, and farthest-point seeding.
- **Manual centroid placement** — click observations in the scene to create the starting configuration you want.
- **13 datasets** — begin with the double helix, then move through clean clouds, ribbons, moons, shells, outliers, and noise.
- **Adjustable experiments** — select 2–6 clusters and generate 160–600 observations.
- **Cluster-wide 3D hulls** — translucent envelopes reveal the full spatial extent of each assigned group.
- **Point inspection** — pin an observation to read its coordinates, assignment, and distance from its centroid.
- **Cluster isolation** — click a cluster in the diagnostics panel or press `1`–`6` to focus on it.
- **Algorithm-aware diagnostics** — follow inertia, medoid cost, negative log-likelihood, density noise, moved observations, representative shift, cluster sizes, and iteration count.
- **Scene controls** — toggle hulls, distance lines, and automatic orbiting independently.
- **Playback tempo** — choose Observe, Flow, or Turbo speed without changing the algorithm.
- **Fresh experiments** — reset an initialization, generate a new sample, or launch a surprise dataset.

## Algorithms

| Algorithm | Model | What the lab exposes |
| --- | --- | --- |
| **K-means** | Nearest-centroid partitions with arithmetic means | Assignment and centroid-update phases, inertia, and centroid movement. |
| **K-medoids** | Nearest-representative partitions using real observations | Medoid selection, total distance cost, and improved resistance to outliers. |
| **DBSCAN** | Density-connected regions with explicit noise | Adjustable neighbourhood radius (`ε`), minimum density, discovered cluster count, and noise observations. |
| **Gaussian mixture model** | Spherical Gaussian components with soft memberships | Expectation/maximization phases, component means and variances, negative log-likelihood, and posterior membership confidence. |

## How the lab is visualized

1. **Choose an algorithm and dataset** to put different clustering assumptions against the same 3D geometry.
2. **Tune the model** with K and an initialization strategy, or DBSCAN's neighbourhood radius and minimum density.
3. **Advance one operation at a time** or run automatically until the selected model completes.
4. **Inspect the result** through spatial hulls, per-cluster counts, objective metrics, noise labels, and point-level details.

K-means and K-medoids deliberately separate assignment from representative updates. The Gaussian mixture model separates its expectation and maximization phases. DBSCAN completes a density expansion in one scan, revealing both connected regions and observations classified as noise. This makes it possible to see *why* each algorithm reaches its result rather than only viewing a final partition.

## Dataset laboratory

| Dataset | Recommended K | What it demonstrates |
| --- | ---: | --- |
| Aurora archipelago | 5 | A spacious, clearly separated reference example. |
| Original eight points | 3 | The source coordinates from the original Python implementation. |
| Gaussian constellations | 4 | Compact, similarly sized groups that suit K-means well. |
| Unequal constellations | 4 | Groups with sharply different density and variance. |
| Anisotropic ribbons | 3 | Rotated, elongated clouds that challenge spherical assumptions. |
| Overlapping currents | 3 | Crossing distributions with ambiguous nearest-centroid boundaries. |
| Interlocking moons | 2 | A classic non-convex failure case. |
| Double helix | 2 | Intertwined manifolds whose topology cannot be preserved by centroid partitions. |
| Island bridges | 3 | Dense groups connected by sparse points that pull the means. |
| Signal lattice | 6 | Nine micro-clusters for experimenting with the choice of K. |
| Concentric shells | 3 | Nested layers with no useful centre-based split. |
| Outlier gravity | 4 | Stable clusters distorted by a small number of extreme observations. |
| No structure | 4 | Uniform noise that K-means must partition despite having no natural groups. |

The curated **Clean split**, **Break it**, and **Stress test** shortcuts provide useful starting points. **Surprise me** selects a different dataset and generates a fresh sample.

## Controls

| Input | Action |
| --- | --- |
| Drag | Orbit the 3D scene |
| Scroll | Zoom in or out |
| Click an observation | Pin and inspect it |
| `Space` | Advance one algorithm operation |
| `A` | Start or pause automatic playback |
| `R` | Reset with a fresh initialization |
| `1`–`6` | Isolate the corresponding cluster |
| `Esc` | Clear the pinned point and cluster focus |

## Technology stack

- **React 19** for the interactive interface and state model.
- **TypeScript** for the dataset, clustering, and scene contracts.
- **Three.js** for geometry, materials, lighting, convex hulls, and the 3D coordinate space.
- **React Three Fiber** for rendering the Three.js scene through React.
- **Drei** for camera controls and HTML scene labels.
- **Tailwind CSS / PostCSS** as part of the styling toolchain, with the product UI authored in custom CSS.
- **Vite + Vinext** for development, production builds, and the React Server Components-compatible runtime.
- **OpenAI Sites / Cloudflare Workers** for the hosted production build.

## Getting started

### Requirements

- Node.js `22.13.0` or newer
- npm

### Install and run

```bash
git clone https://github.com/zmaqutu/k-means-clustering-visualization.git
cd k-means-clustering-visualization
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Production build

```bash
npm run build
npm start
```

### Tests

```bash
npm test
```

The test command creates a production build and runs the rendered-HTML checks.

## Project structure

```text
app/
├── KMeansLab.tsx        # Algorithm, datasets, Three.js scene, and interactions
├── globals.css          # Responsive interface and visual system
├── layout.tsx           # Metadata, fonts, and root layout
└── page.tsx             # Application entry route
public/                  # Social preview and browser assets
readmeAssets/            # Images used by this README
tests/                   # Rendered output checks
.github/workflows/       # GitHub Pages deployment workflow
.openai/hosting.json     # Sites project configuration
```

## Deployment

The repository supports two deployment targets:

- `npm run build` creates the Sites/Vinext production output.
- `npm run build:pages` creates the static GitHub Pages output in `dist-pages`.

Pushes to `main` run `.github/workflows/deploy-pages.yml`. To enable GitHub Pages, select **Settings → Pages → Source → GitHub Actions** in the repository.

## Project lineage

This experience expands the original [machine-learning-k-means-clustering](https://github.com/zmaqutu/machine-learning-k-means-clustering) Python project into a real-time 3D learning environment. The original eight-point dataset remains available in the selector.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a focused feature branch.
3. Make and validate your changes.
4. Open a pull request describing the behavior and visual impact.

Useful contribution areas include new educational datasets, accessible interaction improvements, performance work for larger point clouds, and additional clustering diagnostics.

## Future ideas

- Timeline scrubbing to revisit earlier assignments and centroid positions.
- Side-by-side runs for comparing initialization strategies.
- Exportable experiment snapshots and shareable configurations.
- Per-algorithm parameter presets for each dataset.
- Optional animated walkthrough recordings for each failure-case dataset.

<p align="center">Made with care in React, TypeScript, and Three.js.</p>
