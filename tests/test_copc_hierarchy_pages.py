import io
import struct
import pytest
from forge3d.pointcloud import CopcDataset, OctreeKey, OctreeBounds

def test_follow_pages_and_keep_empty_parents():
    dataset=CopcDataset.__new__(CopcDataset)
    dataset._hierarchy={}
    dataset._info={"root_hier_offset":0,"root_hier_size":64,"spacing":1}
    dataset._root_bounds=OctreeBounds((0,0,0),(4,4,4))
    entry=lambda d,count,offset=0,size=0: struct.pack("<iiiiQii",d,0,0,0,offset,size,count)
    data=entry(0,0)+entry(1,-1,64,64)+entry(1,0)+entry(2,50,128,1)
    dataset._load_hierarchy_pages(io.BytesIO(data+b"x"))
    assert dataset.node_count == 3
    children=dataset.children(OctreeKey.root())
    assert len(children)==1 and children[0].point_count==0
    assert dataset.children(children[0].key)[0].point_count==50
    with pytest.raises(ValueError,match="cyclic"):
        dataset._hierarchy={}
        dataset._load_hierarchy_pages(io.BytesIO(entry(0,-1,0,64)*2))


def test_copc_file_constructor_follows_real_laz_hierarchy():
    from pathlib import Path
    path = Path(__file__).parent / "data" / "copc" / "hierarchy-pages.copc.laz"
    dataset = CopcDataset(path)
    assert dataset._is_copc
    assert dataset._header["point_count"] == 50
    assert dataset.node_count == 3
    assert dataset.root_node().point_count == 0
    parent, = dataset.children(OctreeKey.root())
    assert parent.point_count == 0
    child, = dataset.children(parent.key)
    assert child.point_count == 50 and child.key.depth == 2
