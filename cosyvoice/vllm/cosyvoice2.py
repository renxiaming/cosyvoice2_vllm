"""
Compatibility shim.

The original implementation was committed as `cosyvocie2.py` (typo).
vLLM registration/imports in this repo expect `cosyvoice2.py`.
"""

from .cosyvocie2 import CosyVoice2ForCausalLM  # noqa: F401

