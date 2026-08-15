"use client";

import { Html, Line, OrbitControls, Trail } from "@react-three/drei";
import { Canvas, ThreeEvent, useFrame } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type Vec3 = [number, number, number];
type DatasetId = "classic" | "gaussian" | "overlap" | "shells";
type Strategy = "random" | "plusplus" | "farthest";
type Phase = "ready" | "assigned" | "updated" | "converged";

type PointDatum = {
  id: number;
  position: Vec3;
  cluster: number;
};

type Config = {
  dataset: DatasetId;
  strategy: Strategy;
  k: number;
  pointCount: number;
};

type Model = {
  points: PointDatum[];
  centroids: Vec3[];
  trails: Vec3[][];
  phase: Phase;
  iteration: number;
  moved: number;
  maxShift: number;
  inertia: number;
  events: string[];
};

const CLUSTER_COLORS = ["#76e4f7", "#f6d65f", "#fb7185", "#86efac", "#c4a7ff", "#ff9f66"];
const NEUTRAL_COLOR = "#b7c7d6";
const CLASSIC_POINTS: Vec3[] = [
  [-7.2, 4.8, -1.6], [-7.2, -1.2, 1.2], [7.2, -2.4, -1.1], [0, 3.6, 2.4],
  [4.8, -1.2, 1.6], [2.4, -2.4, -2.2], [-9.6, -6, 0.4], [-2.4, 4.8, -1.2],
];

const DATASETS: Record<DatasetId, { name: string; short: string; note: string }> = {
  classic: {
    name: "Original eight points",
    short: "8-point source set",
    note: "The coordinates from the Python repository, lifted into a shallow third dimension.",
  },
  gaussian: {
    name: "Gaussian constellations",
    short: "Four soft clouds",
    note: "Compact, similarly sized groups—the kind of geometry K-means handles especially well.",
  },
  overlap: {
    name: "Overlapping currents",
    short: "Three stretched clouds",
    note: "Elongated groups cross one another, making the nearest-centroid boundary less obvious.",
  },
  shells: {
    name: "Concentric shells",
    short: "A deliberate failure case",
    note: "Nested spherical layers have no useful centre split, exposing a core limitation of K-means.",
  },
};

const STRATEGIES: Record<Strategy, { label: string; detail: string }> = {
  random: { label: "Random", detail: "Pick K observations directly." },
  plusplus: { label: "K-means++", detail: "Spread seeds probabilistically." },
  farthest: { label: "Farthest", detail: "Always choose the most distant point." },
};

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

function makePoints(dataset: DatasetId, requestedCount: number, seed: number): PointDatum[] {
  if (dataset === "classic") {
    return CLASSIC_POINTS.map((position, id) => ({ id, position, cluster: -1 }));
  }

  const random = mulberry32(seed);
  const points: PointDatum[] = [];
  const count = requestedCount;

  if (dataset === "gaussian") {
    const centers: Vec3[] = [[-7, 2, -5], [7, 3, -4], [-2, -4, 7], [7, -3, 6]];
    for (let id = 0; id < count; id += 1) {
      const center = centers[id % centers.length];
      points.push({
        id,
        cluster: -1,
        position: [
          center[0] + gaussian(random) * 1.75,
          center[1] + gaussian(random) * 1.35,
          center[2] + gaussian(random) * 1.75,
        ],
      });
    }
  } else if (dataset === "overlap") {
    const centers: Vec3[] = [[-4, 1.5, -3], [3, -1, 0], [0, 2, 4]];
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
  } else {
    const radii = [4.2, 8.2, 11.7];
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

function createModel(config: Config, seed: number): Model {
  const points = makePoints(config.dataset, config.pointCount, seed);
  const centroids = initializeCentroids(points, config.k, config.strategy, seed);
  return {
    points,
    centroids,
    trails: centroids.map((centroid) => [[...centroid] as Vec3]),
    phase: "ready",
    iteration: 0,
    moved: 0,
    maxShift: 0,
    inertia: 0,
    events: ["Centroids initialized"],
  };
}

function advanceModel(model: Model): Model {
  if (model.phase === "converged") return model;

  if (model.phase === "ready" || model.phase === "updated") {
    let moved = 0;
    const points = model.points.map((point) => {
      let nearest = 0;
      let nearestDistance = Infinity;
      model.centroids.forEach((centroid, index) => {
        const distance = distanceSquared(point.position, centroid);
        if (distance < nearestDistance) {
          nearest = index;
          nearestDistance = distance;
        }
      });
      if (point.cluster !== nearest) moved += 1;
      return { ...point, cluster: nearest };
    });
    const converged = model.iteration > 0 && moved === 0;
    return {
      ...model,
      points,
      moved,
      inertia: calculateInertia(points, model.centroids),
      phase: converged ? "converged" : "assigned",
      events: [...model.events, converged ? "Assignments unchanged · converged" : `${moved} assignments changed`],
    };
  }

  const sums = model.centroids.map(() => [0, 0, 0, 0]);
  model.points.forEach((point) => {
    const sum = sums[point.cluster];
    sum[0] += point.position[0];
    sum[1] += point.position[1];
    sum[2] += point.position[2];
    sum[3] += 1;
  });
  const centroids = model.centroids.map((centroid, index) => {
    const [x, y, z, count] = sums[index];
    return count > 0 ? [x / count, y / count, z / count] as Vec3 : [...centroid] as Vec3;
  });
  const shifts = centroids.map((centroid, index) => Math.sqrt(distanceSquared(centroid, model.centroids[index])));
  const maxShift = Math.max(...shifts);
  const converged = maxShift <= Number.EPSILON;
  return {
    ...model,
    centroids,
    trails: model.trails.map((trail, index) => [...trail, [...centroids[index]] as Vec3]),
    iteration: model.iteration + 1,
    maxShift,
    inertia: calculateInertia(model.points, centroids),
    phase: converged ? "converged" : "updated",
    events: [...model.events, converged ? "Centroids stationary · converged" : `Centroids moved up to ${maxShift.toFixed(2)} units`],
  };
}

function PointCloud({ points, hovered, onHover }: {
  points: PointDatum[];
  hovered: number | null;
  onHover: (id: number | null) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const glowRef = useRef<THREE.InstancedMesh>(null);
  const currentColors = useRef<THREE.Color[]>([]);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const targetColor = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    currentColors.current = points.map((point) => new THREE.Color(
      point.cluster < 0 ? NEUTRAL_COLOR : CLUSTER_COLORS[point.cluster],
    ));
  }, [points.length, targetColor]);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    const glow = glowRef.current;
    if (!mesh || !glow) return;
    const damping = 1 - Math.exp(-delta * 7);
    points.forEach((point, index) => {
      const isHovered = point.id === hovered;
      dummy.position.set(...point.position);
      const scale = isHovered ? 1.75 : point.cluster < 0 ? 0.88 : 1.08;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      glow.setMatrixAt(index, dummy.matrix);
      targetColor.set(isHovered ? "#ffffff" : point.cluster < 0 ? NEUTRAL_COLOR : CLUSTER_COLORS[point.cluster]);
      if (!currentColors.current[index]) currentColors.current[index] = targetColor.clone();
      currentColors.current[index].lerp(targetColor, damping);
      mesh.setColorAt(index, currentColors.current[index]);
      glow.setColorAt(index, currentColors.current[index]);
    });
    mesh.instanceMatrix.needsUpdate = true;
    glow.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (glow.instanceColor) glow.instanceColor.needsUpdate = true;
  });

  const handlePointer = (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation();
    if (Number.isInteger(event.instanceId)) onHover(points[event.instanceId as number]?.id ?? null);
  };

  return (
    <group>
      <instancedMesh
        ref={glowRef}
        args={[undefined, undefined, points.length]}
        raycast={() => null}
      >
        <sphereGeometry args={[0.4, 10, 10]} />
        <meshBasicMaterial
          vertexColors
          toneMapped={false}
          transparent
          opacity={0.16}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </instancedMesh>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, points.length]}
        onPointerMove={handlePointer}
        onPointerOut={() => onHover(null)}
      >
        <sphereGeometry args={[0.29, 14, 14]} />
        <meshBasicMaterial vertexColors toneMapped={false} />
      </instancedMesh>
    </group>
  );
}

function AnimatedCentroid({ position, index }: { position: Vec3; index: number }) {
  const group = useRef<THREE.Group>(null);
  const target = useMemo(() => new THREE.Vector3(...position), [position]);

  useFrame((_, delta) => {
    if (!group.current) return;
    group.current.position.lerp(target, 1 - Math.exp(-delta * 3.8));
  });

  return (
    <group ref={group} position={position}>
      <Trail
        width={3.4}
        length={12}
        decay={1.15}
        stride={0.012}
        color={CLUSTER_COLORS[index]}
        attenuation={(t) => t * t}
      >
        <mesh raycast={() => null}>
          <sphereGeometry args={[0.22, 12, 12]} />
          <meshBasicMaterial
            color={CLUSTER_COLORS[index]}
            toneMapped={false}
            transparent
            opacity={0.9}
          />
        </mesh>
      </Trail>
      <mesh>
        <octahedronGeometry args={[0.62, 0]} />
        <meshStandardMaterial
          color={CLUSTER_COLORS[index]}
          emissive={CLUSTER_COLORS[index]}
          emissiveIntensity={0.55}
          roughness={0.28}
        />
      </mesh>
      <mesh scale={1.75}>
        <octahedronGeometry args={[0.62, 0]} />
        <meshBasicMaterial color={CLUSTER_COLORS[index]} wireframe transparent opacity={0.42} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.05, 0.025, 8, 48]} />
        <meshBasicMaterial color={CLUSTER_COLORS[index]} transparent opacity={0.7} />
      </mesh>
      <Html position={[0, 1.45, 0]} center zIndexRange={[40, 0]}>
        <div className="centroid-label" style={{ borderColor: `${CLUSTER_COLORS[index]}66` }}>
          C{index + 1}
        </div>
      </Html>
    </group>
  );
}

function ConnectionLines({ points, centroids }: { points: PointDatum[]; centroids: Vec3[] }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    points.forEach((point) => {
      if (point.cluster < 0) return;
      positions.push(...point.position, ...centroids[point.cluster]);
      const color = new THREE.Color(CLUSTER_COLORS[point.cluster]);
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });
    const output = new THREE.BufferGeometry();
    output.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    output.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return output;
  }, [centroids, points]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={0.12} depthWrite={false} />
    </lineSegments>
  );
}

function ClusterHalo({ centroid, radius, index }: { centroid: Vec3; radius: number; index: number }) {
  const group = useRef<THREE.Group>(null);
  const currentRadius = useRef(radius);
  const target = useMemo(() => new THREE.Vector3(...centroid), [centroid]);

  useFrame(({ clock }, delta) => {
    if (!group.current) return;
    group.current.position.lerp(target, 1 - Math.exp(-delta * 3.4));
    currentRadius.current = THREE.MathUtils.damp(currentRadius.current, radius, 3.4, delta);
    const pulse = 1 + Math.sin(clock.elapsedTime * 1.25 + index) * 0.018;
    group.current.scale.setScalar(currentRadius.current * pulse);
    group.current.rotation.y += delta * (0.05 + index * 0.008);
  });

  return (
    <group ref={group} position={centroid} scale={radius}>
      <mesh raycast={() => null}>
        <sphereGeometry args={[1, 28, 18]} />
        <meshBasicMaterial
          color={CLUSTER_COLORS[index]}
          side={THREE.BackSide}
          transparent
          opacity={0.055}
          depthWrite={false}
          toneMapped={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh raycast={() => null}>
        <sphereGeometry args={[1.015, 18, 12]} />
        <meshBasicMaterial
          color={CLUSTER_COLORS[index]}
          wireframe
          transparent
          opacity={0.11}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} raycast={() => null}>
        <torusGeometry args={[1.035, 0.009, 6, 80]} />
        <meshBasicMaterial color={CLUSTER_COLORS[index]} transparent opacity={0.36} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh rotation={[0.72, 0.25, 0.58]} raycast={() => null}>
        <torusGeometry args={[1.05, 0.007, 6, 80]} />
        <meshBasicMaterial color={CLUSTER_COLORS[index]} transparent opacity={0.22} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh rotation={[-0.46, 0.92, -0.2]} raycast={() => null}>
        <torusGeometry args={[1.025, 0.006, 6, 80]} />
        <meshBasicMaterial color={CLUSTER_COLORS[index]} transparent opacity={0.16} depthWrite={false} toneMapped={false} />
      </mesh>
    </group>
  );
}

function ClusterHalos({ points, centroids }: { points: PointDatum[]; centroids: Vec3[] }) {
  return (
    <>
      {centroids.map((centroid, index) => {
        const members = points.filter((point) => point.cluster === index);
        const distances = members
          .map((point) => Math.sqrt(distanceSquared(point.position, centroid)))
          .sort((left, right) => left - right);
        const percentileIndex = Math.min(distances.length - 1, Math.floor(distances.length * 0.82));
        const radius = distances.length ? Math.max(1.45, distances[percentileIndex] + 0.55) : 1.45;
        return (
          <ClusterHalo key={index} centroid={centroid} radius={radius} index={index} />
        );
      })}
    </>
  );
}

function CentroidHistory({ points, index }: { points: Vec3[]; index: number }) {
  if (points.length < 2) return null;
  return (
    <>
      <Line
        points={points}
        color={CLUSTER_COLORS[index]}
        lineWidth={8}
        transparent
        opacity={0.1}
        raycast={() => null}
      />
      <Line
        points={points}
        color={CLUSTER_COLORS[index]}
        lineWidth={2.2}
        transparent
        opacity={0.74}
        raycast={() => null}
      />
      {points.slice(0, -1).map((point, pointIndex) => (
        <mesh key={pointIndex} position={point} raycast={() => null}>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshBasicMaterial color={CLUSTER_COLORS[index]} toneMapped={false} transparent opacity={0.72} />
        </mesh>
      ))}
    </>
  );
}

function Scene({ model, hovered, showLinks, showVolumes, autoRotate, onHover }: {
  model: Model;
  hovered: number | null;
  showLinks: boolean;
  showVolumes: boolean;
  autoRotate: boolean;
  onHover: (id: number | null) => void;
}) {
  const hoveredPoint = hovered === null ? null : model.points.find((point) => point.id === hovered);
  return (
    <>
      <color attach="background" args={["#101820"]} />
      <fog attach="fog" args={["#101820", 35, 74]} />
      <ambientLight intensity={1.35} color="#c7d4df" />
      <directionalLight position={[12, 18, 9]} intensity={2.35} color="#ffffff" />
      <pointLight position={[-12, -4, -10]} intensity={32} color="#76e4f7" />
      <PointCloud points={model.points} hovered={hovered} onHover={onHover} />
      {showLinks && model.phase !== "ready" && <ConnectionLines points={model.points} centroids={model.centroids} />}
      {showVolumes && model.phase !== "ready" && <ClusterHalos points={model.points} centroids={model.centroids} />}
      {model.trails.map((trail, index) => (
        <CentroidHistory key={index} points={trail} index={index} />
      ))}
      {model.centroids.map((centroid, index) => (
        <AnimatedCentroid key={index} position={centroid} index={index} />
      ))}
      {hoveredPoint && (
        <Html position={[hoveredPoint.position[0], hoveredPoint.position[1] + 0.9, hoveredPoint.position[2]]} center zIndexRange={[60, 0]}>
          <div className="point-tooltip">
            <strong>Point {hoveredPoint.id + 1}</strong>
            <span>{hoveredPoint.cluster < 0 ? "Unassigned" : `Cluster ${hoveredPoint.cluster + 1}`}</span>
          </div>
        </Html>
      )}
      <gridHelper args={[42, 21, "#43515f", "#24313d"]} position={[0, -7.5, 0]} />
      <axesHelper args={[12]} position={[0, -7.45, 0]} />
      <Html position={[12.6, -7.35, 0]}><span className="axis-label">X</span></Html>
      <Html position={[0, 5.3, 0]}><span className="axis-label">Y</span></Html>
      <Html position={[0, -7.35, 12.6]}><span className="axis-label">Z</span></Html>
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.42}
        minDistance={16}
        maxDistance={48}
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
  const [config, setConfig] = useState<Config>({ dataset: "gaussian", strategy: "plusplus", k: 4, pointCount: 180 });
  const [seed, setSeed] = useState(1207);
  const [model, setModel] = useState<Model>(() => createModel(config, seed));
  const [isPlaying, setIsPlaying] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [showLinks, setShowLinks] = useState(false);
  const [showVolumes, setShowVolumes] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);

  const step = useCallback(() => setModel((current) => advanceModel(current)), []);

  useEffect(() => {
    if (!isPlaying || model.phase === "converged") {
      if (model.phase === "converged") setIsPlaying(false);
      return undefined;
    }
    const timer = window.setInterval(step, 820);
    return () => window.clearInterval(timer);
  }, [isPlaying, model.phase, step]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "SELECT", "BUTTON"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        step();
      } else if (event.key.toLowerCase() === "a") {
        setIsPlaying((playing) => !playing);
      } else if (event.key.toLowerCase() === "r") {
        setIsPlaying(false);
        setModel(createModel(config, seed));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [config, seed, step]);

  const rebuild = (nextConfig: Config, nextSeed = seed) => {
    setIsPlaying(false);
    setHovered(null);
    setConfig(nextConfig);
    setModel(createModel(nextConfig, nextSeed));
  };

  const changeConfig = <Key extends keyof Config>(key: Key, value: Config[Key]) => {
    rebuild({ ...config, [key]: value });
  };

  const newSample = () => {
    const nextSeed = seed + 137;
    setSeed(nextSeed);
    rebuild(config, nextSeed);
  };

  const clusterCounts = useMemo(() => model.centroids.map((_, index) => (
    model.points.filter((point) => point.cluster === index).length
  )), [model.centroids, model.points]);

  const nextAction = model.phase === "assigned" ? "Update centroids" : "Assign points";
  const status = model.phase === "converged"
    ? { kicker: "System settled", title: "Convergence reached", copy: "Assignments and centroid positions are no longer changing." }
    : model.phase === "assigned"
      ? { kicker: `Iteration ${model.iteration + 1} · Step 2`, title: "Move each centroid", copy: "Replace every centroid with the mean position of the points currently assigned to it." }
      : { kicker: `Iteration ${model.iteration + 1} · Step 1`, title: "Assign every point", copy: "Measure Euclidean distance and give each observation to its nearest centroid." };

  return (
    <main className="lab-shell">
      <div className="scene-layer" aria-label="Interactive three-dimensional K-means visualization">
        <Canvas
          camera={{ position: [22, 17, 25], fov: 48, near: 0.1, far: 120 }}
          dpr={[1, 1.75]}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
          onPointerMissed={() => setHovered(null)}
        >
          <Scene
            model={model}
            hovered={hovered}
            showLinks={showLinks}
            showVolumes={showVolumes}
            autoRotate={autoRotate}
            onHover={setHovered}
          />
        </Canvas>
      </div>

      <section className="control-panel" aria-label="K-means controls">
        <header className="panel-header">
          <div className="brand-mark" aria-hidden="true"><i /><i /><i /></div>
          <div>
            <p className="eyebrow">K-means · 3D lab</p>
            <h1>Watch clusters find their centre.</h1>
          </div>
        </header>
        <p className="lede">Assign points. Update centroids. Repeat until the geometry stops changing.</p>

        <div className="control-group">
          <label htmlFor="dataset">Dataset</label>
          <select id="dataset" value={config.dataset} onChange={(event) => changeConfig("dataset", event.target.value as DatasetId)}>
            {Object.entries(DATASETS).map(([id, dataset]) => <option key={id} value={id}>{dataset.name}</option>)}
          </select>
          <small>{DATASETS[config.dataset].note}</small>
        </div>

        <div className="control-group">
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
        </div>

        <label className="range-control">
          <span><b>Clusters (K)</b><strong>{config.k}</strong></span>
          <input type="range" min="2" max="6" step="1" value={config.k} onChange={(event) => changeConfig("k", Number(event.target.value))} />
        </label>

        <label className={`range-control ${config.dataset === "classic" ? "is-disabled" : ""}`}>
          <span><b>Observations</b><strong>{model.points.length}</strong></span>
          <input
            type="range"
            min="90"
            max="300"
            step="30"
            value={config.pointCount}
            disabled={config.dataset === "classic"}
            onChange={(event) => changeConfig("pointCount", Number(event.target.value))}
          />
        </label>

        <div className="layer-controls" aria-label="Scene layers">
          <Switch checked={showVolumes} onChange={() => setShowVolumes((value) => !value)} label="Group halos" />
          <Switch checked={showLinks} onChange={() => setShowLinks((value) => !value)} label="Distance lines" />
          <Switch checked={autoRotate} onChange={() => setAutoRotate((value) => !value)} label="Auto orbit" />
        </div>

        <button
          type="button"
          className={`auto-run-button ${isPlaying ? "is-running" : ""}`}
          onClick={() => setIsPlaying((playing) => !playing)}
          disabled={model.phase === "converged"}
        >
          <i aria-hidden="true">{isPlaying ? "Ⅱ" : "▶"}</i>
          <span>
            <strong>{isPlaying ? "Pause autoplay" : "Run to convergence"}</strong>
            <small>{isPlaying ? "Following every centroid move" : "Play every step until all means settle"}</small>
          </span>
        </button>

        <div className="secondary-actions">
          <button type="button" onClick={() => rebuild(config)}>Reset</button>
          <button type="button" onClick={newSample}>New sample</button>
        </div>
      </section>

      <div className={`stage-chip ${model.phase === "converged" ? "is-converged" : ""}`} aria-live="polite">
        <span>{status.kicker}</span>
        <strong>{status.title}</strong>
      </div>

      <aside className="insight-rail" aria-label="Algorithm explanation and metrics">
        <section className="insight-card primary-insight">
          <div className="card-heading">
            <span>Current operation</span>
            <b>{model.phase === "assigned" ? "02" : model.phase === "converged" ? "✓" : "01"}</b>
          </div>
          <h2>{status.title}</h2>
          <p>{status.copy}</p>
          <div className="equation">
            {model.phase === "assigned" ? (
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
            <div><span>Inertia</span><strong>{model.phase === "ready" ? "—" : formatMetric(model.inertia)}</strong></div>
            <div><span>Max shift</span><strong>{model.iteration === 0 ? "—" : formatMetric(model.maxShift)}</strong></div>
          </div>
          <div className="cluster-list">
            {model.centroids.map((_, index) => (
              <div key={index}>
                <i style={{ background: CLUSTER_COLORS[index], boxShadow: `0 0 16px ${CLUSTER_COLORS[index]}55` }} />
                <span>Cluster {index + 1}</span>
                <strong>{model.phase === "ready" ? "—" : clusterCounts[index]} pts</strong>
              </div>
            ))}
          </div>
        </section>
      </aside>

      <section className="playback-dock" aria-label="Algorithm playback">
        <button
          type="button"
          className="play-button"
          onClick={() => setIsPlaying((playing) => !playing)}
          disabled={model.phase === "converged"}
          aria-label={isPlaying ? "Pause automatic playback" : "Run automatically"}
        >
          {isPlaying ? "Ⅱ" : "▶"}
        </button>
        <div className="playback-copy">
          <span>{model.phase === "converged" ? "Complete" : isPlaying ? "Autoplay running" : "Next step"}</span>
          <strong>{model.phase === "converged" ? "Exact cluster means found" : isPlaying ? `${nextAction} · until stable` : nextAction}</strong>
        </div>
        <div className="event-track" aria-hidden="true">
          {model.events.slice(-7).map((event, index) => (
            <i key={`${event}-${index}`} className={index === model.events.slice(-7).length - 1 ? "is-current" : ""} />
          ))}
        </div>
        <button type="button" className="step-button" onClick={step} disabled={model.phase === "converged"}>
          Step <span>→</span>
        </button>
      </section>

      <div className="scene-hint">
        <span>Drag to orbit</span><i /> <span>Scroll to zoom</span><i /> <span>Space to step</span>
      </div>
    </main>
  );
}
