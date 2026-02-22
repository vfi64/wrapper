from render_service import RenderService


def test_render_service_delegates_postprocessing_calls():
    calls = []

    def _strip(x):
        calls.append(("strip", x))
        return f"S:{x}"

    def _sanitize(x):
        calls.append(("sanitize", x))
        return f"Z:{x}"

    def _hash(x):
        calls.append(("hash", x))
        return f"H:{x}"

    def _num(x, *, lang="en"):
        calls.append(("num", x, lang))
        return f"N:{x}:{lang}"

    svc = RenderService(
        strip_verification_route_fn=_strip,
        sanitize_self_debunking_html_fn=_sanitize,
        normalize_hash_subheadings_html_fn=_hash,
        number_self_debunking_html_fn=_num,
    )

    assert svc.strip_verification_route_display("a") == "S:a"
    assert svc.sanitize_self_debunking_html("b") == "Z:b"
    assert svc.normalize_hash_subheadings_html("c") == "H:c"
    assert svc.number_self_debunking_html("d", lang="de") == "N:d:de"
    assert [c[0] for c in calls] == ["strip", "sanitize", "hash", "num"]


def test_render_service_is_fail_soft():
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    svc = RenderService(
        strip_verification_route_fn=_boom,
        sanitize_self_debunking_html_fn=_boom,
        normalize_hash_subheadings_html_fn=_boom,
        number_self_debunking_html_fn=_boom,
    )

    assert svc.strip_verification_route_display("x") == "x"
    assert svc.sanitize_self_debunking_html("y") == "y"
    assert svc.normalize_hash_subheadings_html("z") == "z"
    assert svc.number_self_debunking_html("w", lang="de") == "w"
