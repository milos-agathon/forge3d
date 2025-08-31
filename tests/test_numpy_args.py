import numpy as np
import forge3d  # replace with actual module name

def test_signatures_roundtrip():
    ext = np.array([[0.0, 0.0],[1.0, 0.0],[1.0, 1.0],[0.0, 1.0]], dtype=np.float64)
    holes = [np.array([[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]], dtype=np.float64)]
    # functions below should reflect real exported names:
    if hasattr(forge3d, "add_polygons_py"):
        forge3d.add_polygons_py(ext, holes)

    path = np.stack([np.linspace(0,1,5), np.linspace(0,1,5)], axis=1).astype(np.float64)
    if hasattr(forge3d, "add_lines_py"):
        forge3d.add_lines_py(path)

    positions = path.astype(np.float64)
    if hasattr(forge3d, "add_points_py"):
        forge3d.add_points_py(positions)

    nodes = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0]], dtype=np.float64)
    edges = np.array([[0,1],[1,2]], dtype=np.uint32)
    if hasattr(forge3d, "add_graph_py"):
        forge3d.add_graph_py(nodes, edges)