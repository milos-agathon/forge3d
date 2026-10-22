from forge3d.pointcloud import OctreeKey, OctreeBounds, OctreeNode, PointCloudRenderer

class Dataset:
    def root_node(self):
        return OctreeNode(OctreeKey.root(), OctreeBounds((0,0,0),(1,1,1)), 100, 1)
    def children(self, key):
        return []

def test_inside_node_never_overshoots():
    assert PointCloudRenderer(point_budget=1).get_visible_nodes(Dataset(), (0.5,0.5,0.5)) == []
    result = PointCloudRenderer(point_budget=100).get_visible_nodes(Dataset(), (0.5,0.5,0.5))
    assert sum(n.point_count for n in result) == 100
