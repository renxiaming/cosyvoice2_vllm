# SPDX-License-Identifier: Apache-2.0
"""Runtime tweaks for editable / VLLM_TARGET_DEVICE=empty vLLM + vllm_ascend."""


def ensure_vllm_package_has_version() -> None:
    """Set ``vllm.__version__`` when missing.

    vllm_ascend (e.g. ``vllm_version_is`` in worker code) reads ``vllm.__version__``.
    Local ``0.11.0+empty`` trees often omit it on the top-level namespace package.

    If EngineCore still fails after this (e.g. ``spawn`` workers re-import ``vllm``
    from scratch), add a conda ``site-packages/sitecustomize.py`` that calls the
    same logic, or set ``__version__`` in your editable ``vllm/vllm/__init__.py``.
    """
    import vllm as _vp

    if getattr(_vp, "__version__", None):
        return
    try:
        from importlib.metadata import version as _ver

        _vp.__version__ = _ver("vllm")
    except Exception:
        _vp.__version__ = "0.11.0"
