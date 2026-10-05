"use client";

import ProjectCredit from "./ProjectCredit";

import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, ThreeEvent, useFrame } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";

type Vec3 = [number, number, number];
type DatasetId = "showcase" | "classic" | "gaussian" | "varied" | "anisotropic" | "overlap" | "moons" | "helix" | "bridge" | "lattice" | "shells" | "outliers" | "noise";
type Strategy = "random" | "plusplus" | "farthest";
type AlgorithmId = "kmeans" | "kmedoids" | "dbscan" | "gmm";
type Phase = "ready" | "assigned" | "updated" | "converged";
type PlaybackSpeed = "observe" | "normal" | "turbo";

type PointDatum = {
  id: number;
  position: Vec3;
  cluster: number;
};

type Config = {
  algorithm: AlgorithmId;
  dataset: DatasetId;
  strategy: Strategy;
  k: number;
  pointCount: number;
  epsilon: number;
  minPoints: number;
};

type Model = {
  points: PointDatum[];
  centroids: Vec3[];
  phase: Phase;
  iteration: number;
  moved: number;
  maxShift: number;
  inertia: number;
  variances: number[];
  weights: number[];
  responsibilities: number[][];
  events: string[];
};

const CLUSTER_COLORS = ["#76e4f7", "#f6d65f", "#fb7185", "#86efac", "#c4a7ff", "#ff9f66"];
const NEUTRAL_COLOR = "#76e4f7";
const NOISE_COLOR = "#64748b";
const getClusterColor = (index: number) => CLUSTER_COLORS[index % CLUSTER_COLORS.length];
const PHI = (1 + Math.sqrt(5)) / 2;
const CLUSTER_HALO_DIRECTIONS = [
  [0, -1, -PHI], [0, -1, PHI], [0, 1, -PHI], [0, 1, PHI],
  [-1, -PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [1, PHI, 0],
  [-PHI, 0, -1], [-PHI, 0, 1], [PHI, 0, -1], [PHI, 0, 1],
].map(([x, y, z]) => new THREE.Vector3(x, y, z).normalize());
const CLASSIC_POINTS: Vec3[] = [
  [-7.2, 4.8, -1.6], [-7.2, -1.2, 1.2], [7.2, -2.4, -1.1], [0, 3.6, 2.4],
  [4.8, -1.2, 1.6], [2.4, -2.4, -2.2], [-9.6, -6, 0.4], [-2.4, 4.8, -1.2],
];

const DATASETS: Record<DatasetId, { name: string; short: string; note: string; recommendedK: number; badge: string }> = {
  showcase: {
    name: "Aurora archipelago",
    short: "Five wide islands",
    note: "Five crisp constellations spread across the full stage—a spacious, satisfying first run.",
    recommendedK: 5,
    badge: "Showcase",
  },
  classic: {
    name: "Original eight points",
    short: "8-point source set",
    note: "The coordinates from the Python repository, lifted into a shallow third dimension.",
    recommendedK: 3,
    badge: "Original",
  },
  gaussian: {
    name: "Gaussian constellations",
    short: "Four soft clouds",
    note: "Compact, similarly sized groups—the kind of geometry K-means handles especially well.",
    recommendedK: 4,
    badge: "Clear",
  },
  varied: {
    name: "Unequal constellations",
    short: "Mixed density blobs",
    note: "Four clouds with sharply different variances test whether one value of K can describe uneven density.",
    recommendedK: 4,
    badge: "Challenge",
  },
  anisotropic: {
    name: "Anisotropic ribbons",
    short: "Rotated long-form clouds",
    note: "Diagonal, elongated groups expose K-means' preference for compact, spherical clusters.",
    recommendedK: 3,
    badge: "Challenge",
  },
  overlap: {
    name: "Overlapping currents",
    short: "Three stretched clouds",
    note: "Elongated groups cross one another, making the nearest-centroid boundary less obvious.",
    recommendedK: 3,
    badge: "Ambiguous",
  },
  moons: {
    name: "Interlocking moons",
    short: "Non-convex crescents",
    note: "Two curved manifolds make a beautiful failure case: proximity alone cannot preserve their shapes.",
    recommendedK: 2,
    badge: "Failure case",
  },
  helix: {
    name: "Double helix",
    short: "Braided manifolds",
    note: "Two intertwined strands invite you to orbit the scene and watch a centroid method cut across topology.",
    recommendedK: 2,
    badge: "Wild",
  },
  bridge: {
    name: "Island bridges",
    short: "Connected groups",
    note: "Three dense islands connected by sparse causeways reveal how a few points can pull a mean off-centre.",
    recommendedK: 3,
    badge: "Interactive",
  },
  lattice: {
    name: "Signal lattice",
    short: "Nine micro-clusters",
    note: "A precise field of small pods rewards experimenting with K and isolating the resulting regions.",
    recommendedK: 6,
    badge: "Explore K",
  },
  shells: {
    name: "Concentric shells",
    short: "A deliberate failure case",
    note: "Nested spherical layers have no useful centre split, exposing a core limitation of K-means.",
    recommendedK: 3,
    badge: "Failure case",
  },
  outliers: {
    name: "Outlier gravity",
    short: "Clusters plus anomalies",
    note: "Four stable clouds and a handful of distant observations show how strongly means react to extremes.",
    recommendedK: 4,
    badge: "Stress test",
  },
  noise: {
    name: "No structure",
    short: "Uniform null case",
    note: "A homogeneous field has no natural groups, yet K-means must still partition it into Voronoi regions.",
    recommendedK: 4,
    badge: "Null case",
  },
};

const STRATEGIES: Record<Strategy, { label: string; detail: string }> = {
  random: { label: "Random", detail: "Pick K observations directly." },
  plusplus: { label: "K-means++", detail: "Spread seeds probabilistically." },
  farthest: { label: "Farthest", detail: "Always choose the most distant point." },
};

const ALGORITHMS: Record<AlgorithmId, { label: string; short: string; description: string }> = {
  kmeans: {
    label: "K-means",
    short: "Fast centroid partitions",
    description: "Assign points to the nearest mean, then move each mean until the geometry settles.",
  },
  kmedoids: {
    label: "K-medoids",
    short: "Robust point representatives",
    description: "Represent each cluster with a real observation, reducing the pull of extreme outliers.",
  },
  dbscan: {
    label: "DBSCAN",
    short: "Density and noise discovery",
    description: "Grow clusters from dense neighbourhoods and leave isolated observations explicitly marked as noise.",
  },
  gmm: {
    label: "Gaussian mixture",
    short: "Soft probabilistic clusters",
    description: "Alternate expectation and maximization to estimate spherical Gaussian components and membership confidence.",
  },
};

const PLAYBACK_SPEEDS: Record<PlaybackSpeed, { label: string; delay: number }> = {
  observe: { label: "Observe", delay: 1250 },
  normal: { label: "Flow", delay: 760 },
  turbo: { label: "Turbo", delay: 340 },
};

const EXPERIMENT_DECK: { dataset: DatasetId; label: string }[] = [
  { dataset: "showcase", label: "Clean split" },
  { dataset: "helix", label: "Break it" },
  { dataset: "outliers", label: "Stress test" },
];

function mulberry32(seed: number) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random: () => number) {
  const u = Math.max(random(), 1e-8);
  const v = Math.max(random(), 1e-8);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function distanceSquared(a: Vec3, b: Vec3) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function pointPaletteIndex(point: PointDatum) {
  if (point.cluster === -1) return 0;
  if (point.cluster === -2) return 1;
  return 2 + (point.cluster % CLUSTER_COLORS.length);
}

function makePoints(dataset: DatasetId, requestedCount: number, seed: number): PointDatum[] {
  if (dataset === "classic") {
    return CLASSIC_POINTS.map((position, id) => ({ id, position, cluster: -1 }));
  }

  const random = mulberry32(seed);
  const points: PointDatum[] = [];
  const count = requestedCount;

  if (dataset === "showcase") {
    const centers: Vec3[] = [[0, 11, 0], [0, -11, 0], [-13, -4, -11], [13, 5, -11], [0, -5, 14]];
    for (let id = 0; id < count; id += 1) {
      const center = centers[id % centers.length];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + gaussian(random) * 1.25,
          center[1] + gaussian(random) * 1.1,
          center[2] + gaussian(random) * 1.25,
        ],
      });
    }
  } else if (dataset === "gaussian") {
    const centers: Vec3[] = [[-11, -4, -9], [10, 7, -8], [-8, 8, 10], [10, -7, 9]];
    for (let id = 0; id < count; id += 1) {
      const center = centers[id % centers.length];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + gaussian(random) * 1.7,
          center[1] + gaussian(random) * 1.5,
          center[2] + gaussian(random) * 1.7,
        ],
      });
    }
  } else if (dataset === "varied") {
    const centers: Vec3[] = [[-11, -5, -9], [9, 7, -8], [-8, 8, 10], [10, -7, 9]];
    const spreads = [0.8, 2.9, 1.35, 2.05];
    for (let id = 0; id < count; id += 1) {
      const group = id % centers.length;
      const center = centers[group];
      const spread = spreads[group];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + gaussian(random) * spread,
          center[1] + gaussian(random) * spread * 0.82,
          center[2] + gaussian(random) * spread * 1.08,
        ],
      });
    }
  } else if (dataset === "anisotropic") {
    const centers: Vec3[] = [[-9, -5, -9], [7, 6, -2], [2, -6, 10]];
    for (let id = 0; id < count; id += 1) {
      const group = id % centers.length;
      const center = centers[group];
      const a = gaussian(random);
      const b = gaussian(random);
      const c = gaussian(random);
      const transforms = [
        [4.5 * a + 0.5 * b, 1.45 * a + 0.9 * c, 2.5 * a + 0.65 * b],
        [-2.6 * a + 0.7 * c, 3.8 * a + 0.55 * b, 2.15 * a + b],
        [3.4 * a + 0.65 * b, -1.8 * a + 0.7 * c, -3.7 * a + 0.55 * b],
      ][group];
      points.push({
        id,
        cluster: -1,
        position: [center[0] + transforms[0], center[1] + transforms[1], center[2] + transforms[2]],
      });
    }
  } else if (dataset === "overlap") {
    const centers: Vec3[] = [[-7, -3, -6], [6, -2, 1], [-1, 5, 7]];
    for (let id = 0; id < count; id += 1) {
      const group = id % centers.length;
      const center = centers[group];
      const long = gaussian(random) * 4.1;
      const narrowA = gaussian(random) * 1.05;
      const narrowB = gaussian(random) * 1.2;
      const directions: Vec3[] = [[1, 0.22, 0.55], [-0.45, 0.35, 1], [0.7, -0.25, -0.75]];
      const direction = directions[group];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + long * direction[0] + narrowA,
          center[1] + long * direction[1] + narrowB,
          center[2] + long * direction[2] - narrowA * 0.6,
        ],
      });
    }
  } else if (dataset === "moons") {
    for (let id = 0; id < count; id += 1) {
      const group = id % 2;
      const angle = random() * Math.PI;
      const jitter = () => gaussian(random) * 0.48;
      const baseX = group === 0 ? Math.cos(angle) * 8 - 3 : (1 - Math.cos(angle)) * 8 - 3;
      const baseY = group === 0 ? Math.sin(angle) * 6 - 3 : 3 - Math.sin(angle) * 6;
      const baseZ = (angle - Math.PI / 2) * (group === 0 ? 2.2 : -2.2);
      points.push({
        id,
        cluster: -1,
        position: [baseX + jitter(), baseY + jitter(), baseZ + jitter() * 1.6],
      });
    }
  } else if (dataset === "helix") {
    for (let id = 0; id < count; id += 1) {
      const group = id % 2;
      const progress = random();
      const angle = progress * Math.PI * 4 + group * Math.PI;
      const radius = 7.1 + gaussian(random) * 0.38;
      points.push({
        id,
        cluster: -1,
        position: [
          Math.cos(angle) * radius + gaussian(random) * 0.32,
          (progress - 0.5) * 21 + gaussian(random) * 0.38,
          Math.sin(angle) * radius + gaussian(random) * 0.32,
        ],
      });
    }
  } else if (dataset === "bridge") {
    const centers: Vec3[] = [[-12, -5, -8], [0, 7, 0], [12, -5, 8]];
    for (let id = 0; id < count; id += 1) {
      const isBridge = id % 5 === 0;
      if (isBridge) {
        const segment = id % 10 === 0 ? 0 : 1;
        const progress = random();
        const from = centers[segment];
        const to = centers[segment + 1];
        points.push({
          id,
          cluster: -1,
          position: [
            THREE.MathUtils.lerp(from[0], to[0], progress) + gaussian(random) * 0.42,
            THREE.MathUtils.lerp(from[1], to[1], progress) + gaussian(random) * 0.42,
            THREE.MathUtils.lerp(from[2], to[2], progress) + gaussian(random) * 0.42,
          ],
        });
      } else {
        const center = centers[id % centers.length];
        points.push({
          id,
          cluster: -1,
          position: [
            center[0] + gaussian(random) * 1.45,
            center[1] + gaussian(random) * 1.25,
            center[2] + gaussian(random) * 1.45,
          ],
        });
      }
    }
  } else if (dataset === "lattice") {
    const centers: Vec3[] = [];
    for (let row = -1; row <= 1; row += 1) {
      for (let column = -1; column <= 1; column += 1) {
        centers.push([column * 10, (row + column) % 2 === 0 ? 4.5 : -4.5, row * 10]);
      }
    }
    for (let id = 0; id < count; id += 1) {
      const center = centers[id % centers.length];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + gaussian(random) * 0.9,
          center[1] + gaussian(random) * 0.8,
          center[2] + gaussian(random) * 0.9,
        ],
      });
    }
  } else if (dataset === "outliers") {
    const centers: Vec3[] = [[-9, -5, -8], [9, 6, -7], [-8, 7, 9], [9, -6, 9]];
    for (let id = 0; id < count; id += 1) {
      if (id % 17 === 0) {
        points.push({
          id,
          cluster: -1,
          position: [(random() - 0.5) * 34, (random() - 0.5) * 25, (random() - 0.5) * 34],
        });
      } else {
        const center = centers[id % centers.length];
        points.push({
          id,
          cluster: -1,
          position: [
            center[0] + gaussian(random) * 1.35,
            center[1] + gaussian(random) * 1.15,
            center[2] + gaussian(random) * 1.35,
          ],
        });
      }
    }
  } else if (dataset === "noise") {
    for (let id = 0; id < count; id += 1) {
      points.push({
        id,
        cluster: -1,
        position: [(random() - 0.5) * 29, (random() - 0.5) * 21, (random() - 0.5) * 29],
      });
    }
  } else {
    const radii = [5.2, 10.1, 15.1];
    for (let id = 0; id < count; id += 1) {
      const radius = radii[id % radii.length] + gaussian(random) * 0.35;
      const theta = random() * Math.PI * 2;
      const phi = Math.acos(2 * random() - 1);
      points.push({
        id,
        cluster: -1,
        position: [
          radius * Math.sin(phi) * Math.cos(theta),
          radius * Math.cos(phi),
          radius * Math.sin(phi) * Math.sin(theta),
        ],
      });
    }
  }

  return points;
}

function initializeCentroids(points: PointDatum[], k: number, strategy: Strategy, seed: number): Vec3[] {
  const random = mulberry32(seed * 17 + 41);
  const chosen: number[] = [];
  const pickUniqueRandom = () => {
    let index = Math.floor(random() * points.length);
    while (chosen.includes(index)) index = (index + 1) % points.length;
    return index;
  };

  chosen.push(pickUniqueRandom());
  while (chosen.length < Math.min(k, points.length)) {
    const nearestDistances = points.map((point) => Math.min(
      ...chosen.map((index) => distanceSquared(point.position, points[index].position)),
    ));

    if (strategy === "random") {
      chosen.push(pickUniqueRandom());
    } else if (strategy === "farthest") {
      let bestIndex = 0;
      nearestDistances.forEach((distance, index) => {
        if (!chosen.includes(index) && distance > nearestDistances[bestIndex]) bestIndex = index;
      });
      if (chosen.includes(bestIndex)) bestIndex = pickUniqueRandom();
      chosen.push(bestIndex);
    } else {
      const total = nearestDistances.reduce((sum, distance, index) => (
        chosen.includes(index) ? sum : sum + distance
      ), 0);
      let threshold = random() * total;
      let selected = pickUniqueRandom();
      for (let index = 0; index < points.length; index += 1) {
        if (chosen.includes(index)) continue;
        threshold -= nearestDistances[index];
        if (threshold <= 0) {
          selected = index;
          break;
        }
      }
      chosen.push(selected);
    }
  }

  return chosen.map((index) => [...points[index].position] as Vec3);
}

function calculateInertia(points: PointDatum[], centroids: Vec3[]) {
  return points.reduce((sum, point) => (
    point.cluster >= 0 ? sum + distanceSquared(point.position, centroids[point.cluster]) : sum
  ), 0);
}

function calculateMedoidCost(points: PointDatum[], medoids: Vec3[]) {
  return points.reduce((sum, point) => (
    point.cluster >= 0 ? sum + Math.sqrt(distanceSquared(point.position, medoids[point.cluster])) : sum
  ), 0);
}

function assignToNearest(points: PointDatum[], representatives: Vec3[]) {
  let moved = 0;
  const assigned = points.map((point) => {
    let nearest = 0;
    let nearestDistance = Infinity;
    representatives.forEach((representative, index) => {
      const distance = distanceSquared(point.position, representative);
      if (distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    if (point.cluster !== nearest) moved += 1;
    return { ...point, cluster: nearest };
  });
  return { points: assigned, moved };
}

function runDbscan(points: PointDatum[], epsilon: number, minPoints: number) {
  const epsilonSquared = epsilon * epsilon;
  const labels = Array(points.length).fill(-1);
  const visited = Array(points.length).fill(false);
  const neighbours = (index: number) => {
    const nearby: number[] = [];
    points.forEach((candidate, candidateIndex) => {
      if (distanceSquared(points[index].position, candidate.position) <= epsilonSquared) nearby.push(candidateIndex);
    });
    return nearby;
  };

  let clusterCount = 0;
  for (let pointIndex = 0; pointIndex < points.length; pointIndex += 1) {
    if (visited[pointIndex]) continue;
    visited[pointIndex] = true;
    const nearby = neighbours(pointIndex);
    if (nearby.length < minPoints) {
      labels[pointIndex] = -2;
      continue;
    }

    labels[pointIndex] = clusterCount;
    const queue = [...nearby];
    const queued = new Set(queue);
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const candidateIndex = queue[cursor];
      if (!visited[candidateIndex]) {
        visited[candidateIndex] = true;
        const candidateNeighbours = neighbours(candidateIndex);
        if (candidateNeighbours.length >= minPoints) {
          candidateNeighbours.forEach((neighbourIndex) => {
            if (!queued.has(neighbourIndex)) {
              queued.add(neighbourIndex);
              queue.push(neighbourIndex);
            }
          });
        }
      }
      if (labels[candidateIndex] < 0) labels[candidateIndex] = clusterCount;
    }
    clusterCount += 1;
  }

  const assigned = points.map((point, index) => ({ ...point, cluster: labels[index] }));
  const centroids = Array.from({ length: clusterCount }, (_, cluster) => {
    const members = assigned.filter((point) => point.cluster === cluster);
    return members.reduce<Vec3>((sum, point) => [
      sum[0] + point.position[0] / members.length,
      sum[1] + point.position[1] / members.length,
      sum[2] + point.position[2] / members.length,
    ], [0, 0, 0]);
  });
  return { points: assigned, centroids, clusterCount };
}

function expectationStep(model: Model) {
  let negativeLogLikelihood = 0;
  let moved = 0;
  const responsibilities = model.points.map((point) => {
    const logScores = model.centroids.map((centroid, index) => {
      const variance = Math.max(model.variances[index], 0.25);
      return Math.log(Math.max(model.weights[index], 1e-8))
        - 1.5 * Math.log(2 * Math.PI * variance)
        - distanceSquared(point.position, centroid) / (2 * variance);
    });
    const maxScore = Math.max(...logScores);
    const exponentials = logScores.map((score) => Math.exp(score - maxScore));
    const total = exponentials.reduce((sum, score) => sum + score, 0);
    negativeLogLikelihood -= maxScore + Math.log(Math.max(total, 1e-12));
    return exponentials.map((score) => score / Math.max(total, 1e-12));
  });
  const points = model.points.map((point, pointIndex) => {
    const cluster = responsibilities[pointIndex].reduce((best, probability, index, values) => (
      probability > values[best] ? index : best
    ), 0);
    if (point.cluster !== cluster) moved += 1;
    return { ...point, cluster };
  });
  return { points, responsibilities, moved, negativeLogLikelihood };
}

function createModel(config: Config, dataSeed: number, initializationSeed = dataSeed): Model {
  const points = makePoints(config.dataset, config.pointCount, dataSeed);
  const centroids = config.algorithm === "dbscan"
    ? []
    : initializeCentroids(points, config.k, config.strategy, initializationSeed);
  return {
    points,
    centroids,
    phase: "ready",
    iteration: 0,
    moved: 0,
    maxShift: 0,
    inertia: 0,
    variances: centroids.map(() => 16),
    weights: centroids.map(() => 1 / Math.max(centroids.length, 1)),
    responsibilities: [],
    events: [config.algorithm === "dbscan" ? "Density field ready" : "Representatives initialized"],
  };
}

function advanceModel(model: Model, config: Config): Model {
  if (model.phase === "converged") return model;

  if (config.algorithm === "dbscan") {
    const result = runDbscan(model.points, config.epsilon, config.minPoints);
    const noiseCount = result.points.filter((point) => point.cluster === -2).length;
    const moved = result.points.filter((point, index) => point.cluster !== model.points[index].cluster).length;
    return {
      ...model,
      ...result,
      phase: "converged",
      iteration: 1,
      moved,
      maxShift: 0,
      inertia: noiseCount,
      events: [...model.events, `${result.clusterCount} density clusters · ${noiseCount} noise points`],
    };
  }

  if (config.algorithm === "gmm" && (model.phase === "ready" || model.phase === "updated")) {
    const expectation = expectationStep(model);
    return {
      ...model,
      points: expectation.points,
      responsibilities: expectation.responsibilities,
      moved: expectation.moved,
      inertia: expectation.negativeLogLikelihood,
      phase: "assigned",
      events: [...model.events, `Membership probabilities estimated · ${expectation.moved} labels changed`],
    };
  }

  if (model.phase === "ready" || model.phase === "updated") {
    const assignment = assignToNearest(model.points, model.centroids);
    const points = assignment.points;
    const moved = assignment.moved;
    const converged = model.iteration > 0 && moved === 0;
    return {
      ...model,
      points,
      moved,
      inertia: config.algorithm === "kmedoids"
        ? calculateMedoidCost(points, model.centroids)
        : calculateInertia(points, model.centroids),
      phase: converged ? "converged" : "assigned",
      events: [...model.events, converged ? "Assignments unchanged · converged" : `${moved} assignments changed`],
    };
  }

  if (config.algorithm === "gmm") {
    const componentMasses = model.centroids.map((_, cluster) => (
      model.responsibilities.reduce((sum, row) => sum + row[cluster], 0)
    ));
    const centroids = model.centroids.map((centroid, cluster) => {
      const mass = componentMasses[cluster];
      if (mass <= 1e-8) return [...centroid] as Vec3;
      return model.points.reduce<Vec3>((sum, point, pointIndex) => {
        const share = model.responsibilities[pointIndex][cluster] / mass;
        return [
          sum[0] + point.position[0] * share,
          sum[1] + point.position[1] * share,
          sum[2] + point.position[2] * share,
        ];
      }, [0, 0, 0]);
    });
    const variances = centroids.map((centroid, cluster) => {
      const mass = componentMasses[cluster];
      if (mass <= 1e-8) return model.variances[cluster];
      const weightedDistance = model.points.reduce((sum, point, pointIndex) => (
        sum + model.responsibilities[pointIndex][cluster] * distanceSquared(point.position, centroid)
      ), 0);
      return Math.max(weightedDistance / (3 * mass), 0.25);
    });
    const weights = componentMasses.map((mass) => Math.max(mass / model.points.length, 1e-8));
    const shifts = centroids.map((centroid, index) => Math.sqrt(distanceSquared(centroid, model.centroids[index])));
    const maxShift = Math.max(...shifts);
    const converged = maxShift < 0.01;
    return {
      ...model,
      centroids,
      variances,
      weights,
      iteration: model.iteration + 1,
      maxShift,
      phase: converged ? "converged" : "updated",
      events: [...model.events, converged ? "Gaussian components stabilized" : `Component means moved up to ${maxShift.toFixed(2)} units`],
    };
  }

  const centroids = config.algorithm === "kmedoids"
    ? model.centroids.map((medoid, cluster) => {
      const members = model.points.filter((point) => point.cluster === cluster);
      if (members.length === 0) return [...medoid] as Vec3;
      let best = members[0];
      let bestCost = Infinity;
      members.forEach((candidate) => {
        const cost = members.reduce((sum, member) => (
          sum + Math.sqrt(distanceSquared(candidate.position, member.position))
        ), 0);
        if (cost < bestCost) {
          best = candidate;
          bestCost = cost;
        }
      });
      return [...best.position] as Vec3;
    })
    : model.centroids.map((centroid, index) => {
      const members = model.points.filter((point) => point.cluster === index);
      if (members.length === 0) return [...centroid] as Vec3;
      return members.reduce<Vec3>((sum, point) => [
        sum[0] + point.position[0] / members.length,
        sum[1] + point.position[1] / members.length,
        sum[2] + point.position[2] / members.length,
      ], [0, 0, 0]);
    });
  const shifts = centroids.map((centroid, index) => Math.sqrt(distanceSquared(centroid, model.centroids[index])));
  const maxShift = Math.max(...shifts);
  const converged = maxShift < 0.001;
  return {
    ...model,
    centroids,
    iteration: model.iteration + 1,
    maxShift,
    inertia: config.algorithm === "kmedoids"
      ? calculateMedoidCost(model.points, centroids)
      : calculateInertia(model.points, centroids),
    phase: converged ? "converged" : "updated",
    events: [...model.events, converged
      ? `${config.algorithm === "kmedoids" ? "Medoids" : "Centroids"} stationary · converged`
      : `${config.algorithm === "kmedoids" ? "Medoids" : "Centroids"} moved up to ${maxShift.toFixed(2)} units`],
  };
}

function PointCloud({ points, hovered, selected, focusedCluster, onHover, onSelect, onPlace }: {
  points: PointDatum[];
  hovered: number | null;
  selected: number | null;
  focusedCluster: number | null;
  onHover: (id: number | null) => void;
  onSelect: (id: number | null) => void;
  onPlace?: (position: Vec3) => void;
}) {
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const coreGeometry = useMemo(() => new THREE.IcosahedronGeometry(0.3, 1), []);
  const glowGeometry = useMemo(() => new THREE.SphereGeometry(0.62, 12, 10), []);
  const glowMaterial = useMemo(() => new THREE.MeshBasicMaterial({
    color: NEUTRAL_COLOR,
    transparent: true,
    opacity: 0.16,
    depthWrite: false,
    toneMapped: false,
    blending: THREE.AdditiveBlending,
  }), []);
  const materials = useMemo(() => [NEUTRAL_COLOR, NOISE_COLOR, ...CLUSTER_COLORS].map((color, paletteIndex) => new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: paletteIndex === 0 ? 2.2 : 0.82,
    roughness: paletteIndex === 0 ? 0.18 : 0.32,
    metalness: paletteIndex === 0 ? 0.02 : 0.08,
    toneMapped: false,
  })), []);
  const neutralPoints = useMemo(() => points.filter((point) => point.cluster === -1), [points]);
  const neutralGlow = useMemo(() => new THREE.InstancedMesh(
    glowGeometry,
    glowMaterial,
    neutralPoints.length,
  ), [glowGeometry, glowMaterial, neutralPoints.length]);

  const batches = useMemo(() => {
    const groups = Array.from({ length: materials.length }, () => [] as PointDatum[]);
    points.forEach((point) => {
      groups[pointPaletteIndex(point)].push(point);
    });

    return groups.map((group, paletteIndex) => {
      const mesh = new THREE.InstancedMesh(coreGeometry, materials[paletteIndex], group.length);
      mesh.userData.pointIds = group.map((point) => point.id);
      return { mesh, points: group };
    }).filter((batch) => batch.points.length > 0);
  }, [coreGeometry, materials, points]);

  useEffect(() => {
    batches.forEach(({ mesh, points: batchPoints }) => {
      batchPoints.forEach((point, index) => {
        dummy.position.set(...point.position);
        const isDimmed = focusedCluster !== null && point.cluster !== focusedCluster && point.id !== selected;
        const scale = point.id === selected
          ? 1.95
          : point.id === hovered
            ? 1.58
            : point.cluster === -1
              ? 1.14
              : 1.06;
        dummy.scale.setScalar(isDimmed ? scale * 0.24 : scale);
        dummy.updateMatrix();
        mesh.setMatrixAt(index, dummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
    });
  }, [batches, dummy, focusedCluster, hovered, selected]);

  useEffect(() => {
    neutralPoints.forEach((point, index) => {
      dummy.position.set(...point.position);
      dummy.scale.setScalar(point.id === selected ? 1.5 : point.id === hovered ? 1.28 : 1);
      dummy.updateMatrix();
      neutralGlow.setMatrixAt(index, dummy.matrix);
    });
    neutralGlow.instanceMatrix.needsUpdate = true;
  }, [dummy, hovered, neutralGlow, neutralPoints, selected]);

  useEffect(() => () => {
    coreGeometry.dispose();
    glowGeometry.dispose();
    glowMaterial.dispose();
    materials.forEach((material) => material.dispose());
  }, [coreGeometry, glowGeometry, glowMaterial, materials]);

  const handlePointer = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    const pointId = Number.isInteger(event.instanceId)
      ? event.object.userData.pointIds?.[event.instanceId as number]
      : null;
    onHover(pointId ?? null);
  };

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (!Number.isInteger(event.instanceId)) return;
    event.stopPropagation();
    const pointId = event.object.userData.pointIds?.[event.instanceId as number];
    const point = points.find((candidate) => candidate.id === pointId);
    if (!point) return;
    if (onPlace) onPlace(point.position);
    else onSelect(point.id === selected ? null : point.id);
  };

  return (
    <>
      {neutralPoints.length > 0 && <primitive object={neutralGlow} raycast={() => null} />}
      {batches.map(({ mesh }) => (
        <primitive
          key={mesh.uuid}
          object={mesh}
          onPointerMove={handlePointer}
          onPointerOut={() => onHover(null)}
          onClick={handleClick}
        />
      ))}
    </>
  );
}

function AnimatedCentroid({ position, index, label = "C" }: { position: Vec3; index: number; label?: string }) {
  const group = useRef<THREE.Group>(null);
  const initialPosition = useRef<Vec3>([...position]);
  const target = useMemo(() => new THREE.Vector3(...position), [position]);
  const color = getClusterColor(index);

  useFrame((_, delta) => {
    if (!group.current) return;
    group.current.position.lerp(target, 1 - Math.exp(-delta * 4.6));
  });

  return (
    <group ref={group} position={initialPosition.current}>
      <mesh raycast={() => null}>
        <icosahedronGeometry args={[0.58, 1]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <mesh scale={1.62} raycast={() => null}>
        <icosahedronGeometry args={[0.58, 1]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.24} toneMapped={false} />
      </mesh>
      <mesh scale={1.45} raycast={() => null}>
        <sphereGeometry args={[0.72, 16, 12]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.08}
          depthWrite={false}
          toneMapped={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <Html position={[0, 1.45, 0]} center zIndexRange={[40, 0]}>
        <div className="centroid-label" style={{ borderColor: `${color}66` }}>
          {label}{index + 1}
        </div>
      </Html>
    </group>
  );
}

function ConnectionLines({ points, centroids, focusedCluster }: { points: PointDatum[]; centroids: Vec3[]; focusedCluster: number | null }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    points.forEach((point) => {
      if (point.cluster < 0) return;
      if (focusedCluster !== null && point.cluster !== focusedCluster) return;
      positions.push(...point.position, ...centroids[point.cluster]);
      const color = new THREE.Color(getClusterColor(point.cluster));
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });
    const output = new THREE.BufferGeometry();
    output.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    output.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return output;
  }, [centroids, focusedCluster, points]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={0.12} depthWrite={false} />
    </lineSegments>
  );
}

const haloVertexShader = `
  varying vec3 vNormal;
  varying vec3 vViewDirection;

  void main() {
    vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDirection = normalize(-viewPosition.xyz);
    gl_Position = projectionMatrix * viewPosition;
  }
`;

const haloFragmentShader = `
  uniform vec3 uColor;
  varying vec3 vNormal;
  varying vec3 vViewDirection;

  void main() {
    float facing = abs(dot(normalize(vNormal), normalize(vViewDirection)));
    float fresnel = pow(1.0 - facing, 2.25);
    float alpha = 0.018 + fresnel * 0.16;
    gl_FragColor = vec4(uColor, alpha);
  }
`;

function ClusterShell({ members, index }: { members: PointDatum[]; index: number }) {
  const geometry = useMemo(() => {
    if (members.length === 0) return null;

    const padding = 0.9;
    const samples = members.flatMap((point) => {
      const position = new THREE.Vector3(...point.position);
      return CLUSTER_HALO_DIRECTIONS.map((direction) => (
        position.clone().addScaledVector(direction, padding)
      ));
    });
    const hull = new ConvexGeometry(samples);
    const smoothedHull = mergeVertices(hull, 0.001);
    hull.dispose();
    smoothedHull.computeVertexNormals();
    smoothedHull.computeBoundingSphere();
    return smoothedHull;
  }, [members]);

  const material = useMemo(() => (
    new THREE.ShaderMaterial({
      vertexShader: haloVertexShader,
      fragmentShader: haloFragmentShader,
      uniforms: { uColor: { value: new THREE.Color(getClusterColor(index)) } },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
      blending: THREE.AdditiveBlending,
    })
  ), [index]);

  useEffect(() => () => {
    geometry?.dispose();
  }, [geometry]);

  useEffect(() => () => material.dispose(), [material]);

  if (!geometry) return null;

  return (
    <mesh
      geometry={geometry}
      material={material}
      renderOrder={-3}
      frustumCulled={false}
      raycast={() => null}
    />
  );
}

function Scene({ algorithm, model, runId, hovered, selected, focusedCluster, showLinks, showVolumes, autoRotate, onHover, onSelect, onPlace }: {
  algorithm: AlgorithmId;
  model: Model;
  runId: number;
  hovered: number | null;
  selected: number | null;
  focusedCluster: number | null;
  showLinks: boolean;
  showVolumes: boolean;
  autoRotate: boolean;
  onHover: (id: number | null) => void;
  onSelect: (id: number | null) => void;
  onPlace?: (position: Vec3) => void;
}) {
  const hoveredPoint = hovered === null ? null : model.points.find((point) => point.id === hovered);
  const selectedPoint = selected === null ? null : model.points.find((point) => point.id === selected);
  const inspectedPoint = hoveredPoint ?? selectedPoint;
  const clusterMembers = useMemo(() => model.centroids.map((_, index) => (
    model.points.filter((point) => point.cluster === index)
  )), [model.centroids, model.points]);
  return (
    <>
      <color attach="background" args={["#101820"]} />
      <fog attach="fog" args={["#101820", 35, 74]} />
      <ambientLight intensity={1.35} color="#c7d4df" />
      <directionalLight position={[12, 18, 9]} intensity={2.35} color="#ffffff" />
      <pointLight position={[-12, -4, -10]} intensity={32} color="#76e4f7" />
      <PointCloud
        points={model.points}
        hovered={hovered}
        selected={selected}
        focusedCluster={focusedCluster}
        onHover={onHover}
        onSelect={onSelect}
        onPlace={onPlace}
      />
      {showLinks && algorithm !== "dbscan" && model.phase !== "ready" && (
        <ConnectionLines points={model.points} centroids={model.centroids} focusedCluster={focusedCluster} />
      )}
      {showVolumes && model.phase !== "ready" && model.centroids.map((_, index) => (
        focusedCluster === null || focusedCluster === index ? (
        <ClusterShell
          key={`shell-${runId}-${index}`}
          members={clusterMembers[index]}
          index={index}
        />
        ) : null
      ))}
      {algorithm !== "dbscan" && model.centroids.map((centroid, index) => (
        <AnimatedCentroid
          key={`${runId}-${index}`}
          position={centroid}
          index={index}
          label={algorithm === "kmedoids" ? "M" : algorithm === "gmm" ? "G" : "C"}
        />
      ))}
      {inspectedPoint && (
        <Html position={[inspectedPoint.position[0], inspectedPoint.position[1] + 0.9, inspectedPoint.position[2]]} center zIndexRange={[60, 0]}>
          <div className={`point-tooltip ${selectedPoint?.id === inspectedPoint.id ? "is-pinned" : ""}`}>
            <strong>Point {inspectedPoint.id + 1}</strong>
            <span>{inspectedPoint.cluster === -2 ? "Noise" : inspectedPoint.cluster < 0 ? "Unassigned" : `Cluster ${inspectedPoint.cluster + 1}`}</span>
          </div>
        </Html>
      )}
      <gridHelper args={[58, 29, "#43515f", "#24313d"]} position={[0, 0, 0]} />
      <axesHelper args={[16]} position={[0, 0.05, 0]} />
      <Html position={[16.6, 0.15, 0]}><span className="axis-label">X</span></Html>
      <Html position={[0, 16.6, 0]}><span className="axis-label">Y</span></Html>
      <Html position={[0, 0.15, 16.6]}><span className="axis-label">Z</span></Html>
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.42}
        minDistance={18}
        maxDistance={62}
        target={[0, 0, 0]}
      />
    </>
  );
}

function Switch({ checked, onChange, label }: { checked: boolean; onChange: () => void; label: string }) {
  return (
    <button className="layer-switch" type="button" role="switch" aria-checked={checked} onClick={onChange}>
      <span aria-hidden="true"><i /></span>
      {label}
    </button>
  );
}

function formatMetric(value: number) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toFixed(value < 10 ? 2 : 1);
}

export default function KMeansLab() {
  const [config, setConfig] = useState<Config>({
    algorithm: "kmeans",
    dataset: "helix",
    strategy: "plusplus",
    k: 2,
    pointCount: 360,
    epsilon: 2.5,
    minPoints: 6,
  });
  const [seed, setSeed] = useState(1207);
  const [model, setModel] = useState<Model>(() => createModel(config, seed, 4921));
  const [runId, setRunId] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSeeding, setIsSeeding] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [focusedCluster, setFocusedCluster] = useState<number | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>("normal");
  const [showLinks, setShowLinks] = useState(true);
  const [showVolumes, setShowVolumes] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  const [mobilePanel, setMobilePanel] = useState<"controls" | "insights" | null>(null);

  const readyToRun = config.algorithm === "dbscan" || model.centroids.length === config.k;
  const step = useCallback(() => setModel((current) => (
    config.algorithm === "dbscan" || current.centroids.length === config.k ? advanceModel(current, config) : current
  )), [config]);

  useEffect(() => {
    if (!isPlaying || model.phase === "converged" || !readyToRun) {
      if (model.phase === "converged") setIsPlaying(false);
      return undefined;
    }
    const timer = window.setInterval(step, PLAYBACK_SPEEDS[playbackSpeed].delay);
    return () => window.clearInterval(timer);
  }, [isPlaying, model.phase, playbackSpeed, readyToRun, step]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "SELECT", "BUTTON"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        step();
      } else if (event.key.toLowerCase() === "a") {
        if (readyToRun && model.phase !== "converged") {
          setIsSeeding(false);
          setIsPlaying((playing) => !playing);
        }
      } else if (event.key.toLowerCase() === "r") {
        setIsPlaying(false);
        setRunId((current) => current + 1);
        setIsSeeding(false);
        setModel(createModel(config, seed, Math.floor(Math.random() * 1_000_000_000)));
      } else if (event.key === "Escape") {
        setSelected(null);
        setFocusedCluster(null);
        setMobilePanel(null);
      } else if (/^[1-6]$/.test(event.key) && model.phase !== "ready") {
        const clusterIndex = Number(event.key) - 1;
        if (clusterIndex < model.centroids.length) {
          setFocusedCluster((current) => current === clusterIndex ? null : clusterIndex);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [config, model.phase, readyToRun, seed, step]);

  const rebuild = (nextConfig: Config, nextSeed = seed) => {
    setIsPlaying(false);
    setIsSeeding(false);
    setHovered(null);
    setSelected(null);
    setFocusedCluster(null);
    setRunId((current) => current + 1);
    setConfig(nextConfig);
    setModel(createModel(nextConfig, nextSeed, Math.floor(Math.random() * 1_000_000_000)));
  };

  const changeConfig = <Key extends keyof Config>(key: Key, value: Config[Key]) => {
    rebuild({ ...config, [key]: value });
  };

  const selectAlgorithm = (algorithm: AlgorithmId) => {
    rebuild({ ...config, algorithm });
  };

  const selectDataset = (dataset: DatasetId) => {
    rebuild({
      ...config,
      dataset,
      k: DATASETS[dataset].recommendedK,
    });
  };

  const newSample = () => {
    const nextSeed = Math.floor(Math.random() * 1_000_000_000);
    setSeed(nextSeed);
    rebuild(config, nextSeed);
  };

  const surpriseMe = () => {
    const options = (Object.keys(DATASETS) as DatasetId[]).filter((dataset) => dataset !== config.dataset && dataset !== "classic");
    const dataset = options[Math.floor(Math.random() * options.length)];
    const nextSeed = Math.floor(Math.random() * 1_000_000_000);
    setSeed(nextSeed);
    rebuild({ ...config, dataset, k: DATASETS[dataset].recommendedK }, nextSeed);
  };

  const clearForPlacement = () => {
    setIsPlaying(false);
    setIsSeeding(true);
    setHovered(null);
    setSelected(null);
    setFocusedCluster(null);
    setRunId((current) => current + 1);
    setModel((current) => ({
      ...current,
      points: current.points.map((point) => ({ ...point, cluster: -1 })),
      centroids: [],
      phase: "ready",
      iteration: 0,
      moved: 0,
      maxShift: 0,
      inertia: 0,
      variances: [],
      weights: [],
      responsibilities: [],
      events: ["Manual centroid placement started"],
    }));
  };

  const placeCentroid = useCallback((position: Vec3) => {
    if (!isSeeding) return;
    if (model.centroids.length >= config.k) return;
    const duplicate = model.centroids.some((centroid) => distanceSquared(centroid, position) < 0.0001);
    if (duplicate) return;
    const centroids = [...model.centroids, [...position] as Vec3];
    const complete = centroids.length === config.k;
    setIsSeeding(!complete);
    setModel({
      ...model,
      points: model.points.map((point) => ({ ...point, cluster: -1 })),
      centroids,
      phase: "ready",
      iteration: 0,
      moved: 0,
      maxShift: 0,
      inertia: 0,
      variances: centroids.map(() => 16),
      weights: centroids.map(() => 1 / Math.max(centroids.length, 1)),
      responsibilities: [],
      events: [...model.events, complete ? "All manual seeds placed · ready to run" : `Centroid ${centroids.length} placed manually`],
    });
  }, [config.k, isSeeding, model]);

  const addRandomCentroid = () => {
    setIsPlaying(false);
    setHovered(null);
    setSelected(null);
    setFocusedCluster(null);
    if (model.centroids.length >= config.k) setRunId((current) => current + 1);
    const centroids = model.centroids.length >= config.k ? [] : model.centroids;
    const candidates = model.points.filter((point) => (
      !centroids.some((centroid) => distanceSquared(centroid, point.position) < 0.0001)
    ));
    const chosen = candidates[Math.floor(Math.random() * candidates.length)];
    if (!chosen) return;
    const nextCentroids = [...centroids, [...chosen.position] as Vec3];
    const complete = nextCentroids.length === config.k;
    setIsSeeding(!complete);
    setModel({
      ...model,
      points: model.points.map((point) => ({ ...point, cluster: -1 })),
      centroids: nextCentroids,
      phase: "ready",
      iteration: 0,
      moved: 0,
      maxShift: 0,
      inertia: 0,
      variances: nextCentroids.map(() => 16),
      weights: nextCentroids.map(() => 1 / Math.max(nextCentroids.length, 1)),
      responsibilities: [],
      events: [...(centroids.length === 0 ? [] : model.events), complete ? "All random seeds placed · ready to run" : `Centroid ${nextCentroids.length} placed randomly`],
    });
  };

  const togglePlayback = () => {
    if (!readyToRun || model.phase === "converged") return;
    setIsSeeding(false);
    setIsPlaying((playing) => !playing);
  };

  const clusterCounts = useMemo(() => model.centroids.map((_, index) => (
    model.points.filter((point) => point.cluster === index).length
  )), [model.centroids, model.points]);

  const selectedPoint = selected === null ? null : model.points.find((point) => point.id === selected) ?? null;
  const selectedDistance = selectedPoint && selectedPoint.cluster >= 0
    ? Math.sqrt(distanceSquared(selectedPoint.position, model.centroids[selectedPoint.cluster]))
    : null;
  const selectedConfidence = selectedPoint?.cluster !== undefined && selectedPoint.cluster >= 0
    ? model.responsibilities[selectedPoint.id]?.[selectedPoint.cluster] ?? null
    : null;
  const noiseCount = model.points.filter((point) => point.cluster === -2).length;
  const nextAction = config.algorithm === "dbscan"
    ? "Discover density clusters"
    : model.phase === "assigned"
      ? config.algorithm === "gmm" ? "Maximize components" : config.algorithm === "kmedoids" ? "Choose medoids" : "Update centroids"
      : config.algorithm === "gmm" ? "Estimate memberships" : "Assign points";
  const status = (() => {
    if (!readyToRun) return { kicker: "Representative setup", title: `Place seed ${model.centroids.length + 1} of ${config.k}`, copy: "Click an observation to use its exact 3D position, or add a random seed from the controls." };
    if (model.phase === "converged") {
      if (config.algorithm === "dbscan") return { kicker: "Density scan complete", title: `${model.centroids.length} clusters discovered`, copy: `${noiseCount} observations did not belong to a dense region and remain marked as noise.` };
      if (config.algorithm === "gmm") return { kicker: "Model stabilized", title: "Mixture converged", copy: "The Gaussian means, variances, weights, and soft memberships are no longer changing meaningfully." };
      return { kicker: "System settled", title: "Convergence reached", copy: `${config.algorithm === "kmedoids" ? "Medoids" : "Assignments and centroid positions"} are no longer changing.` };
    }
    if (config.algorithm === "dbscan") return { kicker: "Density scan", title: "Discover connected density", copy: "Find core observations, expand through their neighbours, and separate sparse points as noise." };
    if (config.algorithm === "gmm") return model.phase === "assigned"
      ? { kicker: `Iteration ${model.iteration + 1} · M-step`, title: "Fit Gaussian components", copy: "Use soft membership weights to update every component mean, variance, and mixture weight." }
      : { kicker: `Iteration ${model.iteration + 1} · E-step`, title: "Estimate memberships", copy: "Calculate how likely every observation is to belong to each Gaussian component." };
    if (model.phase === "assigned") return config.algorithm === "kmedoids"
      ? { kicker: `Iteration ${model.iteration + 1} · Step 2`, title: "Choose each medoid", copy: "Select the real observation with the lowest total distance to all members of its cluster." }
      : { kicker: `Iteration ${model.iteration + 1} · Step 2`, title: "Move each centroid", copy: "Replace every centroid with the mean position of the points currently assigned to it." };
    return { kicker: `Iteration ${model.iteration + 1} · Step 1`, title: "Assign every point", copy: `Give each observation to its nearest ${config.algorithm === "kmedoids" ? "medoid" : "centroid"}.` };
  })();
  const objectiveLabel = config.algorithm === "kmedoids" ? "Medoid cost" : config.algorithm === "gmm" ? "Neg. log L" : config.algorithm === "dbscan" ? "Noise points" : "Inertia";

  return (
    <main className={`lab-shell ${mobilePanel ? `is-mobile-${mobilePanel}-open` : ""}`}>
      <div className={`scene-layer ${isSeeding ? "is-seeding" : ""}`} aria-label={`Interactive three-dimensional ${ALGORITHMS[config.algorithm].label} visualization`}>
        <Canvas
          camera={{ position: [29, 23, 34], fov: 48, near: 0.1, far: 150 }}
          dpr={[1, 1.75]}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
          onPointerMissed={() => {
            setHovered(null);
            if (!isSeeding) setSelected(null);
          }}
        >
          <Scene
            algorithm={config.algorithm}
            model={model}
            runId={runId}
            hovered={hovered}
            selected={selected}
            focusedCluster={focusedCluster}
            showLinks={showLinks}
            showVolumes={showVolumes}
            autoRotate={autoRotate}
            onHover={setHovered}
            onSelect={setSelected}
            onPlace={isSeeding ? placeCentroid : undefined}
          />
        </Canvas>
        <ProjectCredit />
      </div>

      <button
        type="button"
        className="mobile-panel-backdrop"
        onClick={() => setMobilePanel(null)}
        aria-label="Close mobile panel"
        tabIndex={mobilePanel ? 0 : -1}
      />

      <nav className="mobile-panel-dock" aria-label="Mobile interface panels">
        <button
          type="button"
          className={mobilePanel === "controls" ? "is-active" : ""}
          onClick={() => setMobilePanel((current) => current === "controls" ? null : "controls")}
          aria-expanded={mobilePanel === "controls"}
          aria-controls="clustering-controls"
        >
          <span className="mobile-burger" aria-hidden="true"><i /><i /><i /></span>
          Controls
        </button>
        <button
          type="button"
          className={mobilePanel === "insights" ? "is-active" : ""}
          onClick={() => setMobilePanel((current) => current === "insights" ? null : "insights")}
          aria-expanded={mobilePanel === "insights"}
          aria-controls="clustering-insights"
        >
          <span className="mobile-stats-icon" aria-hidden="true"><i /><i /><i /></span>
          Stats
        </button>
      </nav>

      <section id="clustering-controls" className="control-panel" aria-label="Clustering controls">
        <button type="button" className="mobile-panel-close" onClick={() => setMobilePanel(null)} aria-label="Close controls">×</button>
        <header className="panel-header">
          <div className="brand-mark" aria-hidden="true"><i /><i /><i /></div>
          <div>
            <p className="eyebrow">{ALGORITHMS[config.algorithm].label} · 3D lab</p>
            <h1>Watch structure emerge in 3D.</h1>
          </div>
        </header>
        <p className="lede">{ALGORITHMS[config.algorithm].description}</p>

        <div className="control-group algorithm-control">
          <div className="label-row"><span>Algorithm</span><strong>{ALGORITHMS[config.algorithm].short}</strong></div>
          <div className="algorithm-grid" aria-label="Clustering algorithm">
            {(Object.keys(ALGORITHMS) as AlgorithmId[]).map((algorithm) => (
              <button
                key={algorithm}
                type="button"
                className={config.algorithm === algorithm ? "is-active" : ""}
                onClick={() => selectAlgorithm(algorithm)}
              >
                {ALGORITHMS[algorithm].label}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <div className="label-row"><label htmlFor="dataset">Dataset</label><strong>{DATASETS[config.dataset].badge}</strong></div>
          <select id="dataset" value={config.dataset} onChange={(event) => selectDataset(event.target.value as DatasetId)}>
            {Object.entries(DATASETS).map(([id, dataset]) => <option key={id} value={id}>{dataset.name}</option>)}
          </select>
          <small>{DATASETS[config.dataset].note}</small>
          <div className="experiment-deck" aria-label="Curated experiments">
            {EXPERIMENT_DECK.map((experiment) => (
              <button
                key={experiment.dataset}
                type="button"
                className={config.dataset === experiment.dataset ? "is-active" : ""}
                onClick={() => selectDataset(experiment.dataset)}
              >
                {experiment.label}
              </button>
            ))}
          </div>
        </div>

        {config.algorithm !== "dbscan" && <div className="control-group">
          <div className="label-row"><span>Initialization</span><strong>{STRATEGIES[config.strategy].detail}</strong></div>
          <div className="segmented-control" aria-label="Centroid initialization method">
            {(Object.keys(STRATEGIES) as Strategy[]).map((strategy) => (
              <button
                key={strategy}
                type="button"
                className={config.strategy === strategy ? "is-active" : ""}
                onClick={() => changeConfig("strategy", strategy)}
              >
                {STRATEGIES[strategy].label}
              </button>
            ))}
          </div>
        </div>}

        {config.algorithm !== "dbscan" && <label className="range-control">
          <span><b>Clusters (K)</b><strong>{config.k}</strong></span>
          <input type="range" min="2" max="6" step="1" value={config.k} onChange={(event) => changeConfig("k", Number(event.target.value))} />
        </label>}

        {config.algorithm === "dbscan" && (
          <div className="density-controls">
            <label className="range-control">
              <span><b>Neighbour radius (ε)</b><strong>{config.epsilon.toFixed(2)}</strong></span>
              <input type="range" min="0.75" max="6" step="0.25" value={config.epsilon} onChange={(event) => changeConfig("epsilon", Number(event.target.value))} />
            </label>
            <label className="range-control">
              <span><b>Minimum points</b><strong>{config.minPoints}</strong></span>
              <input type="range" min="3" max="16" step="1" value={config.minPoints} onChange={(event) => changeConfig("minPoints", Number(event.target.value))} />
            </label>
          </div>
        )}

        <label className={`range-control ${config.dataset === "classic" ? "is-disabled" : ""}`}>
          <span><b>Observations</b><strong>{model.points.length}</strong></span>
          <input
            type="range"
            min="160"
            max="600"
            step="40"
            value={config.pointCount}
            disabled={config.dataset === "classic"}
            onChange={(event) => changeConfig("pointCount", Number(event.target.value))}
          />
        </label>

        {config.algorithm !== "dbscan" && <div className={`seed-control ${isSeeding ? "is-active" : ""}`}>
          <div className="label-row">
            <span>{config.algorithm === "kmedoids" ? "Medoid" : config.algorithm === "gmm" ? "Component" : "Centroid"} placement</span>
            <strong>{model.centroids.length} / {config.k} seeds</strong>
          </div>
          <div className="seed-actions">
            <button type="button" onClick={clearForPlacement}>Pick on points</button>
            <button type="button" onClick={addRandomCentroid}>Random <span>＋</span></button>
          </div>
          <small>{isSeeding ? "Select observations in the scene, or add random seeds one at a time." : "Re-seed manually at any time; Reset now produces a fresh automatic initialization."}</small>
        </div>}

        <div className="layer-controls" aria-label="Scene layers">
          <Switch checked={showVolumes} onChange={() => setShowVolumes((value) => !value)} label="3D regions" />
          {config.algorithm !== "dbscan" && <Switch checked={showLinks} onChange={() => setShowLinks((value) => !value)} label={config.algorithm === "gmm" ? "Hard assignment lines" : "Distance lines"} />}
          <Switch checked={autoRotate} onChange={() => setAutoRotate((value) => !value)} label="Auto orbit" />
        </div>

        <div className="tempo-control">
          <div className="label-row"><span>Playback tempo</span><strong>{PLAYBACK_SPEEDS[playbackSpeed].delay} ms</strong></div>
          <div className="segmented-control" aria-label="Autoplay speed">
            {(Object.keys(PLAYBACK_SPEEDS) as PlaybackSpeed[]).map((speed) => (
              <button
                key={speed}
                type="button"
                className={playbackSpeed === speed ? "is-active" : ""}
                onClick={() => setPlaybackSpeed(speed)}
              >
                {PLAYBACK_SPEEDS[speed].label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          className={`auto-run-button ${isPlaying ? "is-running" : ""}`}
          onClick={togglePlayback}
          disabled={model.phase === "converged" || !readyToRun}
        >
          <i aria-hidden="true">{isPlaying ? "Ⅱ" : "▶"}</i>
          <span>
            <strong>{isPlaying ? "Pause autoplay" : "Run to convergence"}</strong>
            <small>{isPlaying ? "Following every algorithm step" : config.algorithm === "dbscan" ? "Scan the complete density field" : "Play every step until the model settles"}</small>
          </span>
        </button>

        <div className="secondary-actions">
          <button type="button" onClick={() => rebuild(config)}>Reset</button>
          <button type="button" onClick={newSample}>New sample</button>
          <button type="button" onClick={surpriseMe}>Surprise me</button>
        </div>
      </section>

      <div className={`stage-chip ${model.phase === "converged" ? "is-converged" : ""}`} aria-live="polite">
        <span>{status.kicker}</span>
        <strong>{status.title}</strong>
      </div>

      <aside id="clustering-insights" className="insight-rail" aria-label="Algorithm explanation and metrics">
        <button type="button" className="mobile-panel-close" onClick={() => setMobilePanel(null)} aria-label="Close diagnostics">×</button>
        <section className="insight-card primary-insight">
          <div className="card-heading">
            <span>Current operation</span>
            <b>{model.phase === "assigned" ? "02" : model.phase === "converged" ? "✓" : "01"}</b>
          </div>
          <h2>{status.title}</h2>
          <p>{status.copy}</p>
          <div className="equation">
            {!readyToRun ? (
              <><span>μ</span><b>←</b><strong>choose an observation</strong></>
            ) : config.algorithm === "dbscan" ? (
              <><span>Nε(x)</span><b>≥</b><strong>minPts</strong></>
            ) : config.algorithm === "gmm" && model.phase === "assigned" ? (
              <><span>μₖ</span><b>←</b><strong>Σ γᵢₖxᵢ / Σ γᵢₖ</strong></>
            ) : config.algorithm === "gmm" ? (
              <><span>γᵢₖ</span><b>∝</b><strong>πₖ N(xᵢ | μₖ, σ²ₖ)</strong></>
            ) : config.algorithm === "kmedoids" && model.phase === "assigned" ? (
              <><span>mⱼ</span><b>=</b><strong>arg min Σ d(xᵢ, m)</strong></>
            ) : model.phase === "assigned" ? (
              <><span>μⱼ</span><b>=</b><strong>Σ xᵢ / |Cⱼ|</strong></>
            ) : (
              <><span>c(xᵢ)</span><b>=</b><strong>arg min ‖xᵢ − μⱼ‖²</strong></>
            )}
          </div>
        </section>

        <section className="metrics-card">
          <div className="card-heading"><span>Live diagnostics</span><b>{DATASETS[config.dataset].short}</b></div>
          <div className="metric-grid">
            <div><span>Iterations</span><strong>{model.iteration}</strong></div>
            <div><span>Points moved</span><strong>{model.phase === "ready" ? "—" : model.moved}</strong></div>
            <div><span>{objectiveLabel}</span><strong>{model.phase === "ready" ? "—" : formatMetric(model.inertia)}</strong></div>
            <div><span>Max shift</span><strong>{model.iteration === 0 ? "—" : formatMetric(model.maxShift)}</strong></div>
          </div>
          <div className="cluster-list">
            {model.centroids.map((_, index) => (
              <button
                key={index}
                type="button"
                className={focusedCluster === index ? "is-focused" : ""}
                onClick={() => setFocusedCluster((current) => current === index ? null : index)}
                disabled={model.phase === "ready"}
                aria-pressed={focusedCluster === index}
                title={`Isolate cluster ${index + 1}`}
              >
                <i style={{ background: getClusterColor(index), boxShadow: `0 0 16px ${getClusterColor(index)}55` }} />
                <span>Cluster {index + 1}</span>
                <strong>{model.phase === "ready" ? "—" : clusterCounts[index]} pts</strong>
              </button>
            ))}
            {config.algorithm === "dbscan" && model.phase !== "ready" && (
              <div className="noise-summary">
                <i style={{ background: NOISE_COLOR }} />
                <span>Noise</span>
                <strong>{noiseCount} pts</strong>
              </div>
            )}
          </div>
        </section>

        {selectedPoint && (
          <section className="insight-card point-inspector" aria-label={`Point ${selectedPoint.id + 1} inspector`}>
            <div className="card-heading">
              <span>Pinned observation</span>
              <button type="button" onClick={() => setSelected(null)} aria-label="Close point inspector">×</button>
            </div>
            <div className="point-inspector-title">
              <i style={{ background: selectedPoint.cluster === -2 ? NOISE_COLOR : selectedPoint.cluster < 0 ? NEUTRAL_COLOR : getClusterColor(selectedPoint.cluster) }} />
              <div><strong>Point {selectedPoint.id + 1}</strong><span>{selectedPoint.cluster === -2 ? "Density noise" : selectedPoint.cluster < 0 ? "Awaiting assignment" : `Cluster ${selectedPoint.cluster + 1}`}</span></div>
            </div>
            <div className="coordinate-grid">
              {selectedPoint.position.map((coordinate, index) => (
                <div key={index}><span>{["X", "Y", "Z"][index]}</span><strong>{coordinate.toFixed(2)}</strong></div>
              ))}
            </div>
            <p>{selectedConfidence !== null
              ? `${(selectedConfidence * 100).toFixed(1)}% posterior membership confidence.`
              : selectedPoint.cluster === -2
                ? "This observation does not have enough density-connected neighbours."
                : selectedDistance === null
                  ? "Run an assignment step to inspect this observation's relationship to the model."
                  : `${selectedDistance.toFixed(2)} units from its current representative.`}</p>
          </section>
        )}
      </aside>

      <section className="playback-dock" aria-label="Algorithm playback">
        <button
          type="button"
          className="play-button"
          onClick={togglePlayback}
          disabled={model.phase === "converged" || !readyToRun}
          aria-label={isPlaying ? "Pause automatic playback" : "Run automatically"}
        >
          {isPlaying ? "Ⅱ" : "▶"}
        </button>
        <div className="playback-copy">
          <span>{!readyToRun ? "Seed setup" : model.phase === "converged" ? "Complete" : isPlaying ? "Autoplay running" : "Next step"}</span>
          <strong>{!readyToRun ? `${model.centroids.length} of ${config.k} representatives placed` : model.phase === "converged" ? config.algorithm === "dbscan" ? `${model.centroids.length} clusters · ${noiseCount} noise` : `${ALGORITHMS[config.algorithm].label} solution found` : isPlaying ? `${nextAction} · until stable` : nextAction}</strong>
        </div>
        <div className="event-track" aria-hidden="true">
          {model.events.slice(-7).map((event, index) => (
            <i key={`${event}-${index}`} className={index === model.events.slice(-7).length - 1 ? "is-current" : ""} />
          ))}
        </div>
        <button type="button" className="step-button" onClick={step} disabled={model.phase === "converged" || !readyToRun}>
          Step <span>→</span>
        </button>
      </section>

      <div className="scene-hint">
        <span>{isSeeding ? "Click a point to seed" : "Click a point to inspect"}</span><i /> <span>Drag to orbit</span><i /> <span>1–6 isolate clusters</span><i /> <span>Space to step</span>
      </div>
    </main>
  );
}
