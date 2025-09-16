# python/forge3d/sdf.py
# Python bindings for SDF (Signed Distance Function) primitives and CSG operations
# Provides high-level interface for creating and rendering SDF scenes

import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
from enum import Enum
import warnings

try:
    from . import forge3d_native  # Rust extension module
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    warnings.warn("Native SDF module not available, using fallback implementation")


class SdfPrimitiveType(Enum):
    """SDF primitive types"""
    SPHERE = 0
    BOX = 1
    CYLINDER = 2
    PLANE = 3
    TORUS = 4
    CAPSULE = 5


class CsgOperation(Enum):
    """CSG (Constructive Solid Geometry) operation types"""
    UNION = 0
    INTERSECTION = 1
    SUBTRACTION = 2
    SMOOTH_UNION = 3
    SMOOTH_INTERSECTION = 4
    SMOOTH_SUBTRACTION = 5


class TraversalMode(Enum):
    """Hybrid traversal modes"""
    HYBRID = 0      # Both SDF and mesh
    SDF_ONLY = 1    # Only SDF geometry
    MESH_ONLY = 2   # Only mesh geometry


class SdfPrimitive:
    """SDF primitive wrapper"""

    def __init__(self, primitive_type: SdfPrimitiveType, center: Tuple[float, float, float],
                 material_id: int, **kwargs):
        """
        Create an SDF primitive

        Args:
            primitive_type: Type of primitive (sphere, box, etc.)
            center: Center position (x, y, z)
            material_id: Material identifier
            **kwargs: Primitive-specific parameters
        """
        self.primitive_type = primitive_type
        self.center = np.array(center, dtype=np.float32)
        self.material_id = material_id
        self.params = kwargs

    @classmethod
    def sphere(cls, center: Tuple[float, float, float], radius: float, material_id: int = 1) -> 'SdfPrimitive':
        """Create a sphere primitive"""
        return cls(SdfPrimitiveType.SPHERE, center, material_id, radius=radius)

    @classmethod
    def box(cls, center: Tuple[float, float, float], extents: Tuple[float, float, float],
            material_id: int = 1) -> 'SdfPrimitive':
        """Create a box primitive"""
        return cls(SdfPrimitiveType.BOX, center, material_id, extents=extents)

    @classmethod
    def cylinder(cls, center: Tuple[float, float, float], radius: float, height: float,
                 material_id: int = 1) -> 'SdfPrimitive':
        """Create a cylinder primitive"""
        return cls(SdfPrimitiveType.CYLINDER, center, material_id, radius=radius, height=height)

    @classmethod
    def plane(cls, normal: Tuple[float, float, float], distance: float,
              material_id: int = 1) -> 'SdfPrimitive':
        """Create a plane primitive"""
        return cls(SdfPrimitiveType.PLANE, (0, 0, 0), material_id, normal=normal, distance=distance)

    @classmethod
    def torus(cls, center: Tuple[float, float, float], major_radius: float, minor_radius: float,
              material_id: int = 1) -> 'SdfPrimitive':
        """Create a torus primitive"""
        return cls(SdfPrimitiveType.TORUS, center, material_id,
                  major_radius=major_radius, minor_radius=minor_radius)

    @classmethod
    def capsule(cls, point_a: Tuple[float, float, float], point_b: Tuple[float, float, float],
                radius: float, material_id: int = 1) -> 'SdfPrimitive':
        """Create a capsule primitive"""
        center = ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2, (point_a[2] + point_b[2]) / 2)
        return cls(SdfPrimitiveType.CAPSULE, center, material_id,
                  point_a=point_a, point_b=point_b, radius=radius)

    def evaluate(self, point: Tuple[float, float, float]) -> float:
        """Evaluate SDF at a point (CPU fallback)"""
        px, py, pz = point
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz

        if self.primitive_type == SdfPrimitiveType.SPHERE:
            r = self.params['radius']
            return np.sqrt(dx*dx + dy*dy + dz*dz) - r

        elif self.primitive_type == SdfPrimitiveType.BOX:
            ex, ey, ez = self.params['extents']
            qx = abs(dx) - ex
            qy = abs(dy) - ey
            qz = abs(dz) - ez
            return (np.sqrt(max(qx, 0)**2 + max(qy, 0)**2 + max(qz, 0)**2) +
                   min(max(max(qx, qy), qz), 0))

        elif self.primitive_type == SdfPrimitiveType.CYLINDER:
            r = self.params['radius']
            h = self.params['height'] / 2
            xz_dist = np.sqrt(dx*dx + dz*dz)
            radial_dist = xz_dist - r
            vertical_dist = abs(dy) - h
            return (np.sqrt(max(radial_dist, 0)**2 + max(vertical_dist, 0)**2) +
                   min(max(radial_dist, vertical_dist), 0))

        elif self.primitive_type == SdfPrimitiveType.PLANE:
            nx, ny, nz = self.params['normal']
            d = self.params['distance']
            return px * nx + py * ny + pz * nz + d

        elif self.primitive_type == SdfPrimitiveType.TORUS:
            major_r = self.params['major_radius']
            minor_r = self.params['minor_radius']
            xz_dist = np.sqrt(dx*dx + dz*dz)
            q_dist = np.sqrt((xz_dist - major_r)**2 + dy*dy)
            return q_dist - minor_r

        elif self.primitive_type == SdfPrimitiveType.CAPSULE:
            ax, ay, az = self.params['point_a']
            bx, by, bz = self.params['point_b']
            r = self.params['radius']
            # Simplified capsule SDF
            pa_x, pa_y, pa_z = px - ax, py - ay, pz - az
            ba_x, ba_y, ba_z = bx - ax, by - ay, bz - az
            h = np.clip(np.dot([pa_x, pa_y, pa_z], [ba_x, ba_y, ba_z]) /
                       np.dot([ba_x, ba_y, ba_z], [ba_x, ba_y, ba_z]), 0, 1)
            closest_x = ax + ba_x * h
            closest_y = ay + ba_y * h
            closest_z = az + ba_z * h
            return np.sqrt((px - closest_x)**2 + (py - closest_y)**2 + (pz - closest_z)**2) - r

        else:
            return float('inf')  # Unknown primitive


class SdfScene:
    """SDF scene containing multiple primitives and CSG operations"""

    def __init__(self):
        self.primitives: List[SdfPrimitive] = []
        self.operations: List[Dict[str, Any]] = []
        self._bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    def add_primitive(self, primitive: SdfPrimitive) -> int:
        """Add a primitive to the scene, returns its index"""
        self.primitives.append(primitive)
        return len(self.primitives) - 1

    def add_operation(self, operation: CsgOperation, left: int, right: int,
                     smoothing: float = 0.0, material_id: int = 0) -> int:
        """Add a CSG operation between two nodes"""
        op_dict = {
            'operation': operation,
            'left': left,
            'right': right,
            'smoothing': smoothing,
            'material_id': material_id
        }
        self.operations.append(op_dict)
        return len(self.operations) - 1 + len(self.primitives)

    def set_bounds(self, min_point: Tuple[float, float, float],
                   max_point: Tuple[float, float, float]):
        """Set scene bounds for optimization"""
        self._bounds = (min_point, max_point)

    def get_bounds(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get scene bounds"""
        return self._bounds

    def evaluate(self, point: Tuple[float, float, float]) -> Tuple[float, int]:
        """
        Evaluate the SDF scene at a point (CPU fallback)

        Returns:
            Tuple of (distance, material_id)
        """
        if not self.primitives:
            return (float('inf'), 0)

        # If no operations, use simple union
        if not self.operations:
            if len(self.primitives) == 1:
                distance = self.primitives[0].evaluate(point)
                return (distance, self.primitives[0].material_id)

            # For multiple primitives, use simple union
            best_distance = float('inf')
            best_material = 0

            for prim in self.primitives:
                distance = prim.evaluate(point)
                if distance < best_distance:
                    best_distance = distance
                    best_material = prim.material_id

            return (best_distance, best_material)

        # Evaluate CSG tree
        return self._evaluate_csg_tree(point)

    def _evaluate_csg_tree(self, point: Tuple[float, float, float]) -> Tuple[float, int]:
        """Evaluate the full CSG tree at a point."""
        # Create evaluation results for all nodes (primitives + operations)
        results = {}

        # Evaluate all primitives first
        for i, prim in enumerate(self.primitives):
            distance = prim.evaluate(point)
            results[i] = (distance, prim.material_id)

        # Evaluate all operations in order
        for i, op in enumerate(self.operations):
            op_id = len(self.primitives) + i
            left_result = results[op['left']]
            right_result = results[op['right']]

            left_dist, left_mat = left_result
            right_dist, right_mat = right_result

            if op['operation'] == CsgOperation.UNION:
                if left_dist < right_dist:
                    result_dist = left_dist
                    result_mat = left_mat
                else:
                    result_dist = right_dist
                    result_mat = right_mat
            elif op['operation'] == CsgOperation.SUBTRACTION:
                # Subtract right from left: max(left, -right)
                result_dist = max(left_dist, -right_dist)
                result_mat = left_mat if result_dist == left_dist else op['material_id']
            elif op['operation'] == CsgOperation.INTERSECTION:
                result_dist = max(left_dist, right_dist)
                result_mat = left_mat if left_dist > right_dist else right_mat
            else:
                # Default to union for unsupported operations
                result_dist = min(left_dist, right_dist)
                result_mat = left_mat if left_dist < right_dist else right_mat

            results[op_id] = (result_dist, result_mat)

        # Return the result of the last operation
        if self.operations:
            last_op_id = len(self.primitives) + len(self.operations) - 1
            return results[last_op_id]
        else:
            # Fallback to simple union
            return self.evaluate(point)

    def primitive_count(self) -> int:
        """Get number of primitives"""
        return len(self.primitives)

    def operation_count(self) -> int:
        """Get number of operations"""
        return len(self.operations)


class SdfSceneBuilder:
    """Builder pattern for constructing SDF scenes"""

    def __init__(self):
        self.scene = SdfScene()

    def add_sphere(self, center: Tuple[float, float, float], radius: float,
                   material_id: int = 1) -> Tuple['SdfSceneBuilder', int]:
        """Add a sphere primitive"""
        primitive = SdfPrimitive.sphere(center, radius, material_id)
        idx = self.scene.add_primitive(primitive)
        return self, idx

    def add_box(self, center: Tuple[float, float, float], extents: Tuple[float, float, float],
                material_id: int = 1) -> Tuple['SdfSceneBuilder', int]:
        """Add a box primitive"""
        primitive = SdfPrimitive.box(center, extents, material_id)
        idx = self.scene.add_primitive(primitive)
        return self, idx

    def add_cylinder(self, center: Tuple[float, float, float], radius: float, height: float,
                     material_id: int = 1) -> Tuple['SdfSceneBuilder', int]:
        """Add a cylinder primitive"""
        primitive = SdfPrimitive.cylinder(center, radius, height, material_id)
        idx = self.scene.add_primitive(primitive)
        return self, idx

    def add_torus(self, center: Tuple[float, float, float], major_radius: float,
                  minor_radius: float, material_id: int = 1) -> Tuple['SdfSceneBuilder', int]:
        """Add a torus primitive"""
        primitive = SdfPrimitive.torus(center, major_radius, minor_radius, material_id)
        idx = self.scene.add_primitive(primitive)
        return self, idx

    def union(self, left: int, right: int, material_id: int = 0) -> Tuple['SdfSceneBuilder', int]:
        """Union two nodes"""
        idx = self.scene.add_operation(CsgOperation.UNION, left, right, 0.0, material_id)
        return self, idx

    def smooth_union(self, left: int, right: int, smoothing: float,
                     material_id: int = 0) -> Tuple['SdfSceneBuilder', int]:
        """Smooth union two nodes"""
        idx = self.scene.add_operation(CsgOperation.SMOOTH_UNION, left, right, smoothing, material_id)
        return self, idx

    def subtract(self, left: int, right: int, material_id: int = 0) -> Tuple['SdfSceneBuilder', int]:
        """Subtract right node from left node"""
        idx = self.scene.add_operation(CsgOperation.SUBTRACTION, left, right, 0.0, material_id)
        return self, idx

    def intersect(self, left: int, right: int, material_id: int = 0) -> Tuple['SdfSceneBuilder', int]:
        """Intersect two nodes"""
        idx = self.scene.add_operation(CsgOperation.INTERSECTION, left, right, 0.0, material_id)
        return self, idx

    def with_bounds(self, min_point: Tuple[float, float, float],
                    max_point: Tuple[float, float, float]) -> 'SdfSceneBuilder':
        """Set scene bounds"""
        self.scene.set_bounds(min_point, max_point)
        return self

    def build(self) -> SdfScene:
        """Build the final scene"""
        return self.scene


class HybridRenderer:
    """Hybrid renderer combining SDF raymarching with mesh BVH traversal"""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.traversal_mode = TraversalMode.HYBRID
        self.early_exit_distance = 0.01
        self.shadow_softness = 4.0

        # Camera parameters
        self.camera_origin = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov_degrees = 45.0
        self.exposure = 1.0

        self._prepare_camera()

    def set_camera(self, origin: Tuple[float, float, float], target: Tuple[float, float, float],
                   up: Tuple[float, float, float] = (0, 1, 0), fov_degrees: float = 45.0):
        """Set camera parameters"""
        self.camera_origin = np.array(origin, dtype=np.float32)
        self.camera_target = np.array(target, dtype=np.float32)
        self.camera_up = np.array(up, dtype=np.float32)
        self.fov_degrees = fov_degrees
        self._prepare_camera()

    def set_traversal_mode(self, mode: TraversalMode):
        """Set traversal mode (hybrid, SDF only, or mesh only)"""
        self.traversal_mode = mode

    def set_performance_params(self, early_exit_distance: float = 0.01, shadow_softness: float = 4.0):
        """Set performance optimization parameters"""
        self.early_exit_distance = early_exit_distance
        self.shadow_softness = shadow_softness

    def _prepare_camera(self):
        """Prepare camera vectors"""
        forward = self.camera_target - self.camera_origin
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        self.camera_forward = forward
        self.camera_right = right
        self.camera_up = up

    def render_sdf_scene(self, scene: SdfScene, spheres: Optional[List] = None) -> np.ndarray:
        """
        Render an SDF scene to RGBA8 image

        Args:
            scene: SDF scene to render
            spheres: Optional legacy spheres for compatibility

        Returns:
            RGBA8 image as numpy array with shape (height, width, 4)
        """
        if NATIVE_AVAILABLE and hasattr(forge3d_native, 'hybrid_render'):
            # Use native implementation if available
            try:
                return self._render_native(scene, spheres or [])
            except Exception as e:
                warnings.warn(f"Native rendering failed: {e}, falling back to CPU")

        # CPU fallback implementation
        return self._render_cpu_fallback(scene, spheres or [])

    def _render_native(self, scene: SdfScene, spheres: List) -> np.ndarray:
        """Render using native Rust implementation"""
        # This would call into the Rust hybrid path tracer
        # For now, return a placeholder
        raise NotImplementedError("Native hybrid rendering not yet implemented")

    def _render_cpu_fallback(self, scene: SdfScene, spheres: List) -> np.ndarray:
        """CPU fallback raymarching implementation"""
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Simple CPU raymarching
        for y in range(self.height):
            for x in range(self.width):
                # Generate ray
                ndc_x = ((x + 0.5) / self.width) * 2.0 - 1.0
                ndc_y = (1.0 - (y + 0.5) / self.height) * 2.0 - 1.0

                aspect = self.width / self.height
                half_h = np.tan(np.radians(self.fov_degrees) * 0.5)
                half_w = aspect * half_h

                ray_dir = np.array([ndc_x * half_w, ndc_y * half_h, -1.0])
                ray_dir = (ray_dir[0] * self.camera_right +
                          ray_dir[1] * self.camera_up +
                          ray_dir[2] * (-self.camera_forward))
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # Raymarch
                t = 0.001
                max_t = 100.0
                max_steps = 64
                min_distance = 0.001

                hit = False
                color = np.array([0.6, 0.7, 0.9])  # Sky color

                for step in range(max_steps):
                    if t >= max_t:
                        break

                    point = self.camera_origin + ray_dir * t
                    distance, material_id = scene.evaluate(tuple(point))

                    if distance < min_distance:
                        # Hit!
                        hit = True
                        # Simple material colors
                        if material_id == 1:
                            color = np.array([0.8, 0.2, 0.2])  # Red
                        elif material_id == 2:
                            color = np.array([0.2, 0.8, 0.2])  # Green
                        elif material_id == 3:
                            color = np.array([0.2, 0.2, 0.8])  # Blue
                        else:
                            color = np.array([0.9, 0.6, 0.3])  # Orange
                        break

                    t += max(abs(distance), min_distance * 0.1)

                # Convert to 8-bit color
                color = np.clip(color * 255, 0, 255).astype(np.uint8)
                image[y, x, :3] = color
                image[y, x, 3] = 255  # Alpha

        return image

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from last render"""
        return {
            'sdf_steps': 0,
            'bvh_nodes_visited': 0,
            'triangle_tests': 0,
            'total_rays': self.width * self.height,
            'sdf_hits': 0,
            'mesh_hits': 0,
            'performance_overhead': 0.0
        }


# Convenience functions
def create_sphere(center: Tuple[float, float, float], radius: float, material_id: int = 1) -> SdfPrimitive:
    """Create a sphere primitive"""
    return SdfPrimitive.sphere(center, radius, material_id)


def create_box(center: Tuple[float, float, float], extents: Tuple[float, float, float],
               material_id: int = 1) -> SdfPrimitive:
    """Create a box primitive"""
    return SdfPrimitive.box(center, extents, material_id)


def create_simple_scene() -> SdfScene:
    """Create a simple test scene with a few primitives"""
    builder = SdfSceneBuilder()

    # Add a sphere and box
    builder, sphere_idx = builder.add_sphere((0, 0, 0), 1.0, 1)
    builder, box_idx = builder.add_box((2, 0, 0), (0.8, 0.8, 0.8), 2)

    # Union them
    builder, union_idx = builder.union(sphere_idx, box_idx)

    return builder.build()


def render_simple_scene(width: int = 512, height: int = 512) -> np.ndarray:
    """Render a simple test scene"""
    scene = create_simple_scene()
    renderer = HybridRenderer(width, height)
    return renderer.render_sdf_scene(scene)


# Example usage
if __name__ == "__main__":
    # Create a simple scene
    scene = create_simple_scene()
    print(f"Scene has {scene.primitive_count()} primitives and {scene.operation_count()} operations")

    # Test evaluation
    distance, material = scene.evaluate((0, 0, 0))
    print(f"Distance at origin: {distance}, material: {material}")

    # Render scene
    print("Rendering scene...")
    image = render_simple_scene(256, 256)
    print(f"Rendered image shape: {image.shape}")