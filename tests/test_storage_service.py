from pathlib import Path

from storage_service import StorageService


def test_storage_service_append_write_exists_and_safe_resolve(tmp_path):
    svc = StorageService()

    p = tmp_path / "a.txt"
    assert svc.append_text(str(p), "one\n") is True
    assert svc.append_text(str(p), "two\n") is True
    assert p.read_text(encoding="utf-8") == "one\ntwo\n"
    assert svc.exists(str(p)) is True

    resolved = svc.safe_resolve_in_dir(str(tmp_path), "a.txt")
    assert isinstance(resolved, str)
    assert Path(resolved).name == "a.txt"


def test_storage_service_write_json_and_fail_soft(tmp_path):
    svc = StorageService()

    payload = {"x": 1, "y": ["a", "b"]}
    p = tmp_path / "data.json"
    assert svc.write_json(str(p), payload, indent=2) is True
    txt = p.read_text(encoding="utf-8")
    assert '"x": 1' in txt
    assert '"y": [' in txt

    # Fail-soft behavior on invalid path/type input.
    assert svc.append_text("/\0", "x") is False
    assert svc.write_json("/\0", payload) is False
    # Traversal names are reduced to basename and stay inside base dir.
    resolved = svc.safe_resolve_in_dir(str(tmp_path), "../x.json")
    assert isinstance(resolved, str)
    assert Path(resolved).parent == tmp_path
    assert Path(resolved).name == "x.json"


def test_storage_service_write_json_creates_parent_dirs_and_lists_json(tmp_path):
    svc = StorageService()
    nested = tmp_path / "nested" / "path" / "a.json"
    assert svc.write_json(str(nested), {"ok": True}) is True
    assert nested.exists()

    (tmp_path / "nested" / "path" / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested" / "path" / "note.txt").write_text("x", encoding="utf-8")

    listed = svc.list_json_filenames(str(tmp_path / "nested" / "path"), limit=10)
    assert listed == ["b.json", "a.json"]


def test_storage_service_read_write_text_and_read_json(tmp_path):
    svc = StorageService()
    txt_path = tmp_path / "x" / "note.txt"
    assert svc.write_text(str(txt_path), "hallo") is True
    assert svc.read_text(str(txt_path)) == "hallo"

    js_path = tmp_path / "x" / "data.json"
    assert svc.write_json(str(js_path), {"ä": 1}, ensure_ascii=False) is True
    payload = svc.read_json(str(js_path))
    assert isinstance(payload, dict)
    assert payload.get("ä") == 1
