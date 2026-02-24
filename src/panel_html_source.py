from __future__ import annotations


def panel_embedded_html(html_panel_embedded, html_panel) -> str:
    if isinstance(html_panel_embedded, str) and html_panel_embedded:
        return html_panel_embedded
    if isinstance(html_panel, str):
        return html_panel
    return ""


def panel_html_source_from_asset_meta(asset_meta, *, has_html: bool) -> str:
    meta = asset_meta if isinstance(asset_meta, dict) else {}
    src = str(meta.get("source") or "embedded")
    if not has_html:
        return "embedded"
    return "external" if src == "external" else "embedded"


def select_panel_html_for_window(
    *,
    force_embedded_html: bool,
    html_panel,
    html_panel_embedded,
    panel_html_asset_meta,
) -> tuple[str, str]:
    if bool(force_embedded_html):
        txt = panel_embedded_html(html_panel_embedded, html_panel)
        return txt, "embedded"

    txt = html_panel if isinstance(html_panel, str) and html_panel else panel_embedded_html(html_panel_embedded, html_panel)
    src = panel_html_source_from_asset_meta(panel_html_asset_meta, has_html=bool(txt))
    return txt, src
