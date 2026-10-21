import pytest
from forge3d.anamnesis import render_sequence

def test_reference_requires_explicit_opt_in():
    with pytest.raises(RuntimeError,match="No frame renderer"):
        render_sequence({},frames=[0],cache=None)
    result=render_sequence({},frames=[0],cache=None,reference_only=True)
    assert result.frame_blobs[0]


def test_reference_result_and_cli_are_labeled(capsys, tmp_path):
    import json
    from forge3d.anamnesis import _main
    result = render_sequence({}, frames=[0], cache=None, reference_only=True)
    assert result.reference_only is True
    real = render_sequence({}, frames=[0], cache=None, render_frame=lambda recipe, frame: b"pixels", render_frame_fingerprint=b"test-renderer", render_frame_context=b"test-inputs")
    assert real.reference_only is False
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    assert _main(["render-sequence", str(recipe), "--cache", str(tmp_path / "cache"), "--frames", "1", "--reference-only"]) == 0
    assert json.loads(capsys.readouterr().out)["reference_only"] is True
