from __future__ import annotations

class OutputResponseModel:
    """Intermediate deterministic model for post-render output processing."""

    def __init__(self, *, html_body: str = "", answer_lang: str = "de", color: str = "off"):
        self.html_body = str(html_body or "")
        self.answer_lang = str(answer_lang or "de")
        self.color = str(color or "off")

    @staticmethod
    def from_values(
        *,
        html_body: str,
        answer_lang: str = "de",
        color: str = "off",
    ) -> "OutputResponseModel":
        lang_raw = str(answer_lang or "").strip().lower()
        lang = "de" if lang_raw.startswith("de") else "en"
        color_raw = str(color or "").strip().lower()
        color_norm = "on" if color_raw == "on" else "off"
        return OutputResponseModel(
            html_body=str(html_body or ""),
            answer_lang=lang,
            color=color_norm,
        )

    def with_html_body(self, html_body: str) -> "OutputResponseModel":
        return OutputResponseModel(
            html_body=str(html_body or ""),
            answer_lang=self.answer_lang,
            color=self.color,
        )
