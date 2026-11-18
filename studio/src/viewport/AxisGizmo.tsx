import { useMemo } from "react";
import * as THREE from "three";

/**
 * Viewport orientation gizmo (the "3D gizmo"), like Blender's navigation widget.
 *
 * It is drawn in SVG from the camera's orientation rather than with a second
 * WebGL context: no extra renderer, no extra draw calls, and the labels stay
 * crisp at any size.
 */

export type Axis = "x" | "y" | "z";

interface AxisSpec {
  axis: Axis;
  label: string;
  colour: string;
  vector: [number, number, number];
}

const AXES: AxisSpec[] = [
  { axis: "x", label: "X", colour: "#e06c62", vector: [1, 0, 0] },
  { axis: "y", label: "Y", colour: "#7fc07a", vector: [0, 1, 0] },
  { axis: "z", label: "Z", colour: "#6f9fe8", vector: [0, 0, 1] },
];

type Quaternion = [number, number, number, number];

/** Rotate `v` by the inverse of `q`: world direction → view space. */
const VIEW_DIRECTION = new THREE.Vector3();
// Object3D.quaternion is (x, y, z, w); three's Vector3 carries the axis math.
function toViewSpace(v: [number, number, number], q: Quaternion): [number, number, number] {
  VIEW_DIRECTION.set(v[0], v[1], v[2]).applyQuaternion(
    new THREE.Quaternion(-q[0], -q[1], -q[2], q[3]),
  );
  return [VIEW_DIRECTION.x, VIEW_DIRECTION.y, VIEW_DIRECTION.z];
}

export function AxisGizmo({
  quaternion,
  onPick,
}: {
  quaternion: Quaternion;
  onPick(axis: Axis): void;
}) {
  const handles = useMemo(() => {
    const radius = 30;
    const centre = 50;
    return AXES.map((spec) => {
      const [x, y, z] = toViewSpace(spec.vector, quaternion);
      const length = Math.hypot(x, y);
      // Keep far axes shorter, so depth reads without hiding the label.
      const scale = 0.55 + 0.45 * Math.max(0, Math.min(1, z * 0.5 + 0.5));
      return {
        ...spec,
        x: centre + (length > 1e-4 ? (x / Math.max(length, 1)) * radius * scale : 0),
        y: centre - (length > 1e-4 ? (y / Math.max(length, 1)) * radius * scale : 0),
        z,
      };
    }).sort((a, b) => a.z - b.z);
  }, [quaternion]);

  return (
    <div className="axis-gizmo" role="group" aria-label="View orientation">
      <svg viewBox="0 0 100 100" width="100" height="100">
        <circle cx="50" cy="50" r="46" className="axis-gizmo__ring" />
        {handles.map((handle) => {
          const dim = handle.z < -0.35;
          return (
            <g
              key={handle.axis}
              className="axis-gizmo__axis"
              onClick={() => onPick(handle.axis)}
              role="button"
              aria-label={`Look along the ${handle.label} axis`}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onPick(handle.axis);
              }}
            >
              <line
                x1="50"
                y1="50"
                x2={handle.x}
                y2={handle.y}
                stroke={handle.colour}
                strokeWidth={1.6}
                opacity={dim ? 0.45 : 0.95}
              />
              <circle
                cx={handle.x}
                cy={handle.y}
                r={8}
                fill={dim ? "var(--bg-panel-alt)" : handle.colour}
                stroke={handle.colour}
                strokeWidth={1.4}
                opacity={dim ? 0.9 : 1}
              />
              <text
                x={handle.x}
                y={handle.y + 3.2}
                textAnchor="middle"
                fontSize="9"
                fontWeight="600"
                fill={dim ? handle.colour : "#10130f"}
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                {handle.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
