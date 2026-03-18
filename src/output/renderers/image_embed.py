from __future__ import annotations

import html
import re


_IMG_URL_RE = re.compile(
    r"https?://[^\s<>()\]\[]+?\.(?:png|jpe?g|gif|webp|svg)(?:[?#][^\s<>()\]\[]*)?[.,;:!?)>]*",
    re.IGNORECASE,
)
_TRAILING_PUNCT = ".,;:!?)>"
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def _split_trailing_punctuation(token: str) -> tuple[str, str]:
    text = str(token or "")
    trail = ""
    while text and text[-1] in _TRAILING_PUNCT:
        trail = text[-1] + trail
        text = text[:-1]
    return text, trail


def _is_image_url(url: str) -> bool:
    return bool(_IMG_URL_RE.fullmatch(str(url or "")))


def _img_tag(url: str) -> str:
    safe_url = html.escape(str(url or ""), quote=True)
    return (
        "\n\n"
        f'<img src="{safe_url}" style="max-width:100%; height:auto; border-radius:10px; margin:6px 0;" loading="lazy" />'
        "\n"
    )


def _extract_single_image_url_from_fence(fence_body: str) -> str:
    body = str(fence_body or "")
    if not body.strip():
        return ""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    if (
        len(lines) >= 2
        and re.fullmatch(r"[A-Za-z0-9_+\-]{1,24}", lines[0] or "")
        and "://" not in lines[0]
    ):
        lines = lines[1:]
    if len(lines) != 1:
        return ""
    core, _trail = _split_trailing_punctuation(lines[0])
    return core if _is_image_url(core) else ""


def auto_embed_image_urls(text: str) -> str:
    """Embed plain image URLs as inline <img> tags outside fenced code blocks."""
    src = str(text or "")
    if not src or "http" not in src:
        return src

    parts = src.split("```")

    for i in range(0, len(parts), 2):
        seg = str(parts[i] or "")
        protected: list[str] = []
        append_after_token: dict[str, str] = {}

        def _protect_inline_code(m: re.Match) -> str:
            token = f"__IMG_INLINE_CODE_{len(protected)}__"
            full = str(m.group(0) or "")
            inner = str(m.group(1) or "").strip()
            core, _trail = _split_trailing_punctuation(inner)
            if _is_image_url(core):
                append_after_token[token] = _img_tag(core)
            protected.append(full)
            return token

        stage = _INLINE_CODE_RE.sub(_protect_inline_code, seg)

        def _repl_img(m: re.Match) -> str:
            matched = str(m.group(0) or "")
            matched, trail = _split_trailing_punctuation(matched)
            url = matched
            if not url:
                return str(m.group(0) or "")
            img = _img_tag(url)
            return url + trail + img

        stage = _IMG_URL_RE.sub(_repl_img, stage)
        for idx, original in enumerate(protected):
            token = f"__IMG_INLINE_CODE_{idx}__"
            stage = stage.replace(token, original + append_after_token.get(token, ""))
        parts[i] = stage

    # Conservative fence fallback:
    # If a fenced block contains exactly one image URL, keep the fence untouched
    # and append one image preview directly after the closing fence.
    for i in range(1, len(parts), 2):
        fenced_url = _extract_single_image_url_from_fence(parts[i])
        if not fenced_url:
            continue
        if i + 1 >= len(parts):
            parts.append("")
        parts[i + 1] = _img_tag(fenced_url) + str(parts[i + 1] or "")

    return "```".join(parts)
