/**
 * glTF/GLB loading for the Studio viewport.
 *
 * Bytes come from the shell (`read_mesh_file`) or from an in-flight import.
 * Parsed scenes are cached per path: a project can reference the same asset
 * from several objects, and re-parsing a GLB on every state change would make
 * the viewport unusable.
 */

import type { Group, Material, Object3D } from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const bytesByPath = new Map<string, ArrayBuffer>();
const scenesByPath = new Map<string, Promise<Group>>();
const nodesByPath = new Map<string, Promise<string[]>>();

/** Decode the base64 payload the shell returns for a mesh file. */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export function putAssetBytes(path: string, buffer: ArrayBuffer): void {
  bytesByPath.set(path, buffer);
  // Bounded: relinking many assets must not keep every file's bytes alive.
  while (bytesByPath.size > 64) {
    const oldest = bytesByPath.keys().next();
    if (oldest.done) break;
    bytesByPath.delete(oldest.value);
  }
  scenesByPath.delete(path);
  nodesByPath.delete(path);
}

/** Drop the parse caches. Call when the project closes: scene objects are
 *  disposed by the viewport, and these maps would otherwise keep them alive. */
export function clearAssetCaches(): void {
  bytesByPath.clear();
  scenesByPath.clear();
  nodesByPath.clear();
}

function parseGltf(buffer: ArrayBuffer, path: string): Promise<Group> {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    // GLB is self-contained. A .gltf with external buffers cannot be resolved
    // from the webview, so the loader fails loudly and we translate that into
    // advice rather than a stack trace.
    loader.parse(
      buffer,
      "",
      (gltf) => resolve(gltf.scene),
      (error) => {
        const message = error instanceof Error ? error.message : String(error);
        reject(
          new Error(
            `${path}: ${message}. Self-contained .glb files are supported; a .gltf that references external .bin or texture files is not yet.`,
          ),
        );
      },
    );
  });
}

/** Parse without keeping anything: used to validate a file before importing.
 *  Also reports the named, mesh-bearing nodes so the Inspector can offer a
 *  per-object selection. */
export async function inspectGltf(
  buffer: ArrayBuffer,
  path: string,
): Promise<{ meshes: number; nodes: string[] }> {
  const scene = await parseGltf(buffer, path);
  const { meshes, nodes } = inventoryGltf(scene);
  disposeTree(scene);
  if (meshes === 0) {
    throw new Error(`${path} contains no meshes to import`);
  }
  return { meshes, nodes };
}

/** Mesh count plus the names of nodes whose subtree carries a mesh. */
function inventoryGltf(scene: Group): { meshes: number; nodes: string[] } {
  let meshes = 0;
  const nodes = new Set<string>();
  scene.traverse((object) => {
    if (!(object as { isMesh?: boolean }).isMesh) return;
    meshes += 1;
    // Offer every NAMED ancestor between the mesh and the scene: picking a
    // leaf selects one submesh, picking a parent group the whole assembly —
    // the granularity glTF authoring exposes.
    let node: Object3D | null = object;
    while (node && node !== scene) {
      if (node.name) nodes.add(node.name);
      node = node.parent;
    }
    if (scene.name) nodes.add(scene.name);
  });
  return { meshes, nodes: [...nodes].sort() };
}

/** Named mesh-bearing nodes of an asset, cached by path (the Inspector shows
 *  them when the user picks which node an imported object draws). */
export function listGltfNodes(
  path: string,
  read: (path: string) => Promise<ArrayBuffer>,
): Promise<string[]> {
  let pending = nodesByPath.get(path);
  if (!pending) {
    pending = loadSceneGroup(path, read).then((scene) => inventoryGltf(scene).nodes);
    nodesByPath.set(path, pending);
    pending.catch(() => nodesByPath.delete(path));
  }
  return pending;
}

function loadSceneGroup(
  path: string,
  read: (path: string) => Promise<ArrayBuffer>,
): Promise<Group> {
  let pending = scenesByPath.get(path);
  if (!pending) {
    const bytes = bytesByPath.get(path);
    pending = bytes ? parseGltf(bytes, path) : read(path).then((buffer) => parseGltf(buffer, path));
    scenesByPath.set(path, pending);
    pending.catch(() => scenesByPath.delete(path));
  }
  return pending;
}

/**
 * Load an asset, either from the import cache or through `read` (the shell).
 * Returns a clone, so two objects can use the same asset independently. With
 * `node`, only that node's subtree is returned — the unused rest of the clone
 * goes to the JS GC untouched (its geometry/materials are shared with the
 * cache and must not be disposed).
 */
export async function loadAssetScene(
  path: string,
  read: (path: string) => Promise<ArrayBuffer>,
  node?: string | null,
): Promise<Object3D> {
  const scene = await loadSceneGroup(path, read);
  const clone = scene.clone(true);
  if (!node) return clone;
  const chosen = clone.getObjectByName(node);
  if (!chosen) {
    throw new Error(
      `${path}: node "${node}" no longer exists — re-pick it in the object's Mesh section.`,
    );
  }
  return chosen;
}

export function disposeTree(root: Object3D | null): void {
  root?.traverse((object) => {
    const mesh = object as { geometry?: { dispose(): void }; material?: Material | Material[] };
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
    else material?.dispose();
  });
}
