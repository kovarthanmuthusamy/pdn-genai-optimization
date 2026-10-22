import pytest
from pathlib import Path


def test_need_names_the_setting(tmp_path):
    from thesis_code.common.paths import need
    with pytest.raises(SystemExit, match="DATA_DIR"):
        need(tmp_path / "missing", "DATA_DIR")


def test_out_dir_is_under_output_dir(out_dir):
    from thesis_code.common.paths import out_dir as stage_dir
    p = stage_dir("s10_search")
    assert p == out_dir / "s10_search" and p.is_dir()


def test_result_path_prefers_new_output(out_dir):
    from thesis_code.common import settings
    from thesis_code.common.paths import result_path
    assert result_path("ch4_search.json") == settings.SAVED_RESULTS_DIR / "ch4_search.json"
    new = out_dir / "s12_evaluation" / "ch4_search.json"
    new.parent.mkdir(parents=True)
    new.write_text("{}")
    assert result_path("ch4_search.json") == new


def test_result_fallback_is_announced(out_dir, capsys):
    """A check that reads another check's result from the saved results (not made in this output
    folder) says so, so results of another model are not mixed in silently."""
    from thesis_code.common.paths import result_path
    result_path("ch4_accuracy.npz")
    assert "saved result" in capsys.readouterr().out
    new = out_dir / "s12_evaluation" / "ch4_accuracy.npz"
    new.parent.mkdir(parents=True)
    new.write_bytes(b"")
    result_path("ch4_accuracy.npz")
    assert "saved result" not in capsys.readouterr().out


def test_settings_point_to_existing_inputs():
    from thesis_code.common import settings
    for name in ("DATA_DIR", "WEIGHTS_DIR", "INDEPENDENT_DIR", "SAVED_RESULTS_DIR", "SIM_STORE_DIR", "PLAN_DIR"):
        assert getattr(settings, name).exists(), name
    assert settings.ECADSTAR_RUN is False


def test_guard_detects_a_write(tmp_path):
    from tests.thesis_code.conftest import _snapshot
    f = tmp_path / "x.txt"
    f.write_text("a")
    before = _snapshot([tmp_path], recursive=True)
    f.write_text("bb")
    assert _snapshot([tmp_path], recursive=True) != before
