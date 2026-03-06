from __future__ import annotations

import json

import help_i18n as sut


def test_normalize_lang_maps_supported_values():
    assert sut.normalize_lang("de") == "de"
    assert sut.normalize_lang("de-DE") == "de"
    assert sut.normalize_lang("en") == "en"
    assert sut.normalize_lang("en-US") == "en"
    assert sut.normalize_lang("fr") == "de"
    assert sut.normalize_lang("") == "de"


def test_load_help_payload_reads_localized_assets():
    payload = sut.load_help_payload("de")
    assert isinstance(payload, dict)
    assert isinstance(payload.get("sections"), list)
    assert payload.get("title")
    assert payload.get("button_label")
    assert payload.get("close_label")


def test_load_help_payload_falls_back_to_en_when_primary_invalid(tmp_path):
    bad_de = tmp_path / "help.de.json"
    bad_de.write_text("{not-json", encoding="utf-8")

    en_payload = {
        "button_label": "? Help",
        "title": "Fallback Help",
        "close_label": "Close",
        "sections": [
            {"id": "quickstart", "title": "Quickstart", "items": ["A", "B"]},
        ],
    }
    (tmp_path / "help.en.json").write_text(json.dumps(en_payload), encoding="utf-8")

    out = sut.load_help_payload("de", base_dir=tmp_path)
    assert out.get("title") == "Fallback Help"
    assert out.get("button_label") == "? Help"

