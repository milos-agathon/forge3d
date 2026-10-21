import json
from forge3d.pointcloud import EptDataset, PointCloudRenderer

def test_parent_and_child_are_distinct_contributions(tmp_path):
    path = tmp_path / "ept.json"
    path.write_text(json.dumps({"bounds": [0,0,0,1,1,1], "points": 6, "span": 128}))
    hierarchy = tmp_path / "ept-hierarchy"
    hierarchy.mkdir()
    (hierarchy / "0-0-0-0.json").write_text(json.dumps({"0-0-0-0":5,"1-0-0-0":1}))
    dataset = EptDataset(path)
    selected = PointCloudRenderer(point_budget=6).get_visible_nodes(dataset, (0.5,0.5,0.5))
    assert {str(n.key) for n in selected} == {"0-0-0-0", "1-0-0-0"}
    assert sum(n.point_count for n in selected) == 6
    selected = PointCloudRenderer(point_budget=5).get_visible_nodes(dataset, (0.5,0.5,0.5))
    assert sum(n.point_count for n in selected) <= 5
