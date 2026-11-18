/**
 * Polygon meshes for the viewport overlay.
 *
 * The still fills polygons with the engine's polygon rasterizer on the
 * frame; the viewport is a live preview, so it rebuilds the same rings as a
 * draped triangle mesh. Colour, opacity, ring and hole semantics come from
 * the project layer; earcut does the triangulation (THREE.ShapeUtils).
 */
import * as THREE from "three";

import type { HeightPreview } from "./cartography.ts";
import { worldPosition } from "./cartography.ts";

export interface PolygonMeshData {
  positions: Float32Array;
  indices: Uint32Array;
}

/** Signed area of a ring in the lattice's own 2D frame (u, v). */
function signedArea(ring: [number, number][]): number {
  let area = 0;
  for (let index = 0; index < ring.length; index += 1) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    area += x0 * y1 - x1 * y0;
  }
  return area / 2;
}

/**
 * Triangulate a layer's polygons (exterior + holes) onto the terrain.
 *
 * Ring windings are normalised first (exterior CCW, holes CW) so the
 * tessellator gets the orientation it expects regardless of the file's own
 * winding.
 */
export function buildPolygonGeometry(
  exteriors: [number, number][][],
  holes: [number, number][][][],
  preview: HeightPreview,
  spanXz: [number, number],
  zScale: number,
  lift: number,
): PolygonMeshData {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let feature = 0; feature < exteriors.length; feature += 1) {
    const exterior = exteriors[feature];
    if (!exterior || exterior.length < 3) continue;
    const contour = [...exterior];
    if (signedArea(contour) < 0) contour.reverse();
    const holesOfFeature = (holes[feature] ?? [])
      .filter((ring) => ring && ring.length >= 3)
      .map((ring) => (signedArea(ring) > 0 ? [...ring].reverse() : ring));
    const contourVecs = contour.map(([u, v]) => new THREE.Vector2(u, v));
    const holeVecs = holesOfFeature.map((ring) => ring.map(([u, v]) => new THREE.Vector2(u, v)));
    const triangles = THREE.ShapeUtils.triangulateShape(contourVecs, holeVecs);
    if (triangles.length === 0) continue;
    const base = positions.length / 3;
    // Input order is preserved: contour vertices first, then the hole rings
    // in order — triangulateShape's indices refer to that order.
    const all = [...contour, ...holesOfFeature.flat()];
    for (const [u, v] of all) {
      const [x, y, z] = worldPosition(preview, spanXz, zScale, u, v, lift);
      positions.push(x, y, z);
    }
    for (const triangle of triangles) {
      indices.push(base + triangle[0], base + triangle[1], base + triangle[2]);
    }
  }
  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };
}

/** One mesh per polygon layer: colour/opacity from the layer, one draw call. */
export function buildPolygonMesh(
  exteriors: [number, number][][],
  holes: [number, number][][][],
  preview: HeightPreview,
  spanXz: [number, number],
  zScale: number,
  lift: number,
  colour: [number, number, number, number],
): THREE.Mesh | null {
  const { positions, indices } = buildPolygonGeometry(
    exteriors,
    holes,
    preview,
    spanXz,
    zScale,
    lift,
  );
  if (positions.length === 0) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colour[0], colour[1], colour[2]),
    roughness: 0.95,
    metalness: 0.0,
    transparent: colour[3] < 1,
    opacity: colour[3],
    side: THREE.DoubleSide,
    // It is a drape, not a rival surface: lose the depth fight politely.
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 3;
  mesh.frustumCulled = false;
  return mesh;
}
