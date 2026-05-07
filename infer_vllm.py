# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import multiprocessing
import os
import sys

# Ascend/torch_npu: NPU must not be re-initialized in a forked child. vLLM V1
# EngineCore uses multiprocessing; Linux defaults to "fork". Force "spawn"
# before importing torch_npu or starting CosyVoice (ACL / device init).
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# A repo-local top-level `vllm/` directory shadows the pip `vllm` package (same
# name on sys.path) and breaks vllm_ascend: `from vllm import ModelRegistry`.
# Prefer environment site-packages when that layout is detected.
_infer_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_infer_dir, "vllm")):
    import site

    for _sp in reversed(site.getsitepackages()):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)


def _check_vllm_python_deps_or_exit() -> None:
    """Fail fast before heavy imports (WETEXT / OM): vLLM 0.11 needs tokenizers>=0.22."""
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import parse as p

    try:
        if p(version("tokenizers")) < p("0.22.0"):
            print(
                "Error: tokenizers>=0.22.0 required for vLLM 0.11 (DecodeStream in tokenizers.decoders).",
                'Fix: pip install -U "tokenizers>=0.22.0" "transformers>=4.55.2" "huggingface_hub>=0.24.0,<1.0"',
                "Or: pip install -r requirements-vllm-npu.txt",
                file=sys.stderr,
                sep="\n",
            )
            raise SystemExit(1)
        if p(version("transformers")) < p("4.55.2"):
            print(
                "Error: transformers>=4.55.2 required (older transformers caps tokenizers<0.20).",
                'Fix: pip install -U "transformers>=4.55.2" "huggingface_hub>=0.24.0,<1.0"',
                "Or: pip install -r requirements-vllm-npu.txt",
                file=sys.stderr,
                sep="\n",
            )
            raise SystemExit(1)
        if p(version("huggingface_hub")) < p("0.24.0"):
            print(
                "Error: huggingface_hub>=0.24 required with transformers>=4.55 "
                "(hub API used by transformers.utils.hub).",
                'Fix: pip install -U "huggingface_hub>=0.24.0,<1.0"',
                "Or: pip install -r requirements-vllm-npu.txt",
                file=sys.stderr,
                sep="\n",
            )
            raise SystemExit(1)
    except PackageNotFoundError:
        pass


_check_vllm_python_deps_or_exit()

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# vLLM custom model registration.
# NOTE: In vLLM >= 0.11, ModelRegistry may no longer be exported at the top-level
# `vllm` package (especially with `VLLM_TARGET_DEVICE=empty`). Import it from
# the internal registry module with fallbacks for older layouts.
import vllm
from cosyvoice.utils.vllm_runtime import ensure_vllm_package_has_version

ensure_vllm_package_has_version()
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry = None
_model_registry_import_errors = []
for _mod, _attr in (
    ("vllm.model_executor.models.registry", "ModelRegistry"),
    ("vllm.model_executor.models", "ModelRegistry"),
    ("vllm", "ModelRegistry"),
):
    try:
        _m = __import__(_mod, fromlist=[_attr])
        ModelRegistry = getattr(_m, _attr)
        break
    except Exception as e:  # pragma: no cover
        _model_registry_import_errors.append(f"{_mod}.{_attr}: {type(e).__name__}: {e}")

if ModelRegistry is None:
    raise ImportError(
        "Failed to import vLLM ModelRegistry. This usually means your installed "
        "`vllm` package layout doesn't match this integration, or `vllm` isn't "
        "installed in the current environment.\n"
        "Tried:\n- "
        + "\n- ".join(_model_registry_import_errors)
        + "\n\n"
        "Expected setup on Ascend:\n"
        "- Install `vllm` from source with `VLLM_TARGET_DEVICE=empty`\n"
        "- Install `vllm-ascend==0.11.0`\n"
    )

if not hasattr(ModelRegistry, "register_model"):
    raise ImportError(
        f"Imported ModelRegistry from vLLM but it has no `register_model`. "
        f"Got: {ModelRegistry!r}. Your vLLM version may be incompatible."
    )

if getattr(vllm, "ModelRegistry", None) is None:
    setattr(vllm, "ModelRegistry", ModelRegistry)

ensure_vllm_package_has_version()
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=2, type=int, help='warm up times')
    parser.add_argument('--infer_count', default=20, type=int, help='infer loop count')
    parser.add_argument('--stream', action="store_true", help='stream infer')
    args = parser.parse_args()

    cosyvoice = CosyVoice2(args.model_path, load_om=True, fp16=True,load_vllm=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm() #删除推理过程中的weight_norm
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)


    # 输入数据加载
    prompt_txt = '收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'

    with torch.no_grad():
        # import ipdb;ipdb.set_trace()
        print('warm up start')
        for _ in range(args.warm_up_times):
            next(cosyvoice.inference_sft(prompt_txt, '中文女', stream=args.stream))
        print('warm up end')
        # import ipdb;ipdb.set_trace()
        for _ in range(args.infer_count):
            os.makedirs("testout", exist_ok=True)
            if args.stream:
                chunks = []
                for j in cosyvoice.inference_sft(prompt_txt, '中文女', stream=args.stream):
                    tts = j.get('tts_speech')
                    if tts is not None:
                        chunks.append(tts.detach().cpu())
                if not chunks:
                    raise RuntimeError("No audio chunks returned from streaming inference.")
                full = torch.cat(chunks, dim=-1)
                torchaudio.save('testout/sft_vllm.wav', full, cosyvoice.sample_rate)
            else:
                j = next(cosyvoice.inference_sft(prompt_txt, '中文女', stream=args.stream))
                torchaudio.save('testout/sft_vllm.wav', j['tts_speech'].detach().cpu(), cosyvoice.sample_rate)