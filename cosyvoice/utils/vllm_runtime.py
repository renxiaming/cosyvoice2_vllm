# SPDX-License-Identifier: Apache-2.0
"""Runtime tweaks for editable / VLLM_TARGET_DEVICE=empty vLLM + vllm_ascend."""


def assert_vllm_ipc_supports_prompt_embeds() -> None:
    """``EngineCoreRequest`` must include ``prompt_embeds`` (vLLM >= 0.11 with V1 Engine)."""
    from vllm.v1.engine import EngineCoreRequest

    fields = getattr(EngineCoreRequest, "__struct_fields__", ())
    if "prompt_embeds" not in fields:
        raise RuntimeError(
            "Installed vLLM's EngineCoreRequest has no prompt_embeds field "
            f"(fields={fields!r}). Use vLLM 0.11.x V1 engine with "
            "enable_prompt_embeds; mixed old vllm + new vllm_ascend loses embeddings "
            "over IPC and fails in CachedRequestState."
        )


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
