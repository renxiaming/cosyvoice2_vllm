# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from packaging.version import parse as _vparse

from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm


def _require_tokenizers_for_vllm_011() -> None:
    """vLLM 0.11 arg_utils 依赖 tokenizers.decoders.DecodeStream（需 tokenizers>=0.22）。"""
    try:
        if _vparse(_pkg_version("tokenizers")) < _vparse("0.22.0"):
            raise ImportError(
                'tokenizers>=0.22.0 required (DecodeStream). Run: pip install -U '
                '"tokenizers>=0.22.0" "transformers>=4.55.2" '
                "or: pip install -r requirements-vllm-npu.txt"
            )
    except PackageNotFoundError:
        pass


def _import_vllm_engine_classes():
    """Import EngineArgs / LLMEngine for vLLM 0.11.x and vllm-ascend.

    `from vllm import EngineArgs` fails with namespace / partial installs
    (ImportError: ... '(unknown location)'). Submodules resolve reliably.
    """
    _require_tokenizers_for_vllm_011()
    import vllm  # noqa: F401 — ensure top-level package is loaded

    from cosyvoice.utils.vllm_runtime import ensure_vllm_package_has_version

    ensure_vllm_package_has_version()

    errors: list[str] = []
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine

        ensure_vllm_package_has_version()
        return EngineArgs, LLMEngine
    except ImportError as e:
        errors.append(f"vllm.engine.arg_utils / vllm.engine.llm_engine: {e}")
    try:
        from vllm import EngineArgs, LLMEngine  # type: ignore

        ensure_vllm_package_has_version()
        return EngineArgs, LLMEngine
    except ImportError as e:
        errors.append(f"vllm top-level: {e}")
    joined = "\n- ".join(errors)
    hint = ""
    if "DecodeStream" in joined:
        hint = (
            "\n\n(DecodeStream → upgrade tokenizers + transformers: "
            'pip install -U "tokenizers>=0.22.0" "transformers>=4.55.2" '
            "or pip install -r requirements-vllm-npu.txt)\n"
        )
    raise ImportError(
        "Cannot import vLLM EngineArgs / LLMEngine. On Ascend 910B use "
        "vllm-ascend matching your vLLM version (e.g. vllm-ascend==0.11.0). "
        "Tried:\n- " + joined + hint
    ) from None


def _ensure_vllm_top_level_model_registry() -> None:
    """vllm_ascend does ``from vllm import ModelRegistry`` during EngineArgs init.

    A repo-local ``vllm/`` tree or lazy namespace ``vllm`` may not expose it; patch
    the resolved package so plugin import succeeds.
    """
    import vllm as _vp

    if getattr(_vp, "ModelRegistry", None) is not None:
        return
    try:
        from vllm.model_executor.models import ModelRegistry as _MR
    except ImportError:
        from vllm.model_executor.models.registry import ModelRegistry as _MR  # type: ignore
    setattr(_vp, "ModelRegistry", _MR)


def _get_accelerator_device() -> torch.device:
    """Prefer NPU (Ascend) then CUDA, otherwise CPU."""
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def _make_stream_context(device: torch.device):
    if torch.cuda.is_available():
        return torch.cuda.stream(torch.cuda.Stream(device))
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            # torch-npu provides torch.npu.Stream / torch.npu.stream
            return torch.npu.stream(torch.npu.Stream(device))
    except Exception:
        pass
    return nullcontext()


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = _get_accelerator_device()
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # here we fix set flow.decoder.estimator.static_chunk_size = 0 for compatibability
        self.flow.decoder.estimator.static_chunk_size = 0
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = _make_stream_context(self.device)
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model), 'streaming input text is only implemented for CosyVoice2!'
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                for i in self.llm.inference(text=text.to(self.device),
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, flow_cache = self.flow.inference(token=token.to(self.device),
                                                  token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_token=prompt_token.to(self.device),
                                                  prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_feat=prompt_feat.to(self.device),
                                                  prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                  embedding=embedding.to(self.device),
                                                  flow_cache=self.flow_cache_dict[uuid])
        self.flow_cache_dict[uuid] = flow_cache

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.01) # 0.1
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:

                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)


                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        _empty_cache()

    def vc(self, source_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        _empty_cache()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = _get_accelerator_device()
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_hop_len = 1 * self.flow.input_frame_rate #2
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 1 * self.flow.input_frame_rate #2
        self.flow.decoder.estimator.static_chunk_size = 1 * self.flow.input_frame_rate * self.flow.token_mel_ratio #2
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = _make_stream_context(self.device)
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.first_chunk_size = 20

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        from cosyvoice.utils.vllm_runtime import ensure_vllm_package_has_version

        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        EngineArgs, LLMEngine = _import_vllm_engine_classes()
        _ensure_vllm_top_level_model_registry()
        # vLLM / vllm-ascend layouts differ: only pass device= when EngineArgs accepts it.
        base_kwargs = dict(
            model=model_dir,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            gpu_memory_utilization=0.2,
        )
        _ea_init = getattr(EngineArgs, "__init__", None)
        _params = (
            inspect.signature(_ea_init).parameters if _ea_init is not None else {}
        )
        if "device" in _params:
            engine_args = EngineArgs(**base_kwargs, device="npu")
        else:
            engine_args = EngineArgs(**base_kwargs)
        ensure_vllm_package_has_version()
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers
        
    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        
        import time
        # --- Flow 模块计步 ---
        torch.npu.synchronize() # 确保之前的操作完成
        start_flow = time.time()
        
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        
        torch.npu.synchronize()
        flow_time = (time.time() - start_flow) * 1000 # 毫秒

        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]

        # --- Hift (Vocoder) 模块计步 ---
        start_hift = time.time()

        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        
        torch.npu.synchronize()
        hift_time = (time.time() - start_hift) * 1000 # 毫秒
        print(f"模块耗时详情: Flow={flow_time:.2f}ms | Hift={hift_time:.2f}ms")
        
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        
        import time # 确保导入了 time
        import torch_npu
        
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if stream is True:
            token_offset = 0
            
            # 1. 记录 LLM 开始迭代的时间
            llm_start_time = time.time()

            # 删除线程操作，串行执行推理，加速首包时延
            for i in self.llm.inference(text=text.to(self.device),
                                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_text=prompt_text.to(self.device),
                                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=llm_embedding.to(self.device)):
                self.tts_speech_token_dict[this_uuid].append(i)
                if (token_offset == 0 and len(self.tts_speech_token_dict[this_uuid]) >= self.first_chunk_size + self.flow.pre_lookahead_len) or (token_offset > 0 and len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len):
                #if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    #this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    
                    # --- 计算 LLM 攒够这一包 token 的耗时 ---
                    torch_npu.npu.synchronize()
                    llm_duration = (time.time() - llm_start_time) * 1000
                    
                    if token_offset == 0:
                        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:self.first_chunk_size + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    else: 
                        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    
                    # --- 2. 测量 token2wav (包含 Flow 和 Hift) 的耗时 ---
                    start_t2w = time.time()
                    
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                        prompt_token=flow_prompt_speech_token,
                                                        prompt_feat=prompt_speech_feat,
                                                        embedding=flow_embedding,
                                                        uuid=this_uuid,
                                                        token_offset=token_offset,
                                                        finalize=False)
                    
                    torch_npu.npu.synchronize()
                    t2w_duration = (time.time() - start_t2w) * 1000
                    
                    # 打印当前分包的详细耗时
                    print(f"\n[Profiling] Chunk Offset: {token_offset}")
                    print(f" > LLM 累积耗时: {llm_duration:.2f}ms")
                    print(f" > Flow+Hift 推理耗时: {t2w_duration:.2f}ms")
                    
                    #token_offset += self.token_hop_len
                    if token_offset == 0:
                        token_offset += self.first_chunk_size
                    else:
                        token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}

                    # 重置 LLM 计时器，准备下一包
                    llm_start_time = time.time()

            # --- 2. 结尾 Finalize 处理 (此处即为你问的“结尾”) ---
            start_final = time.time()

            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                prompt_token=flow_prompt_speech_token,
                                                prompt_feat=prompt_speech_feat,
                                                embedding=flow_embedding,
                                                uuid=this_uuid,
                                                token_offset=token_offset,
                                                finalize=True)
            
            torch_npu.npu.synchronize()
            final_duration = (time.time() - start_final) * 1000
            print(f"[Profiling Final] Tail Logic Time: {final_duration:.1f}ms")

            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            p.start()
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
        _empty_cache()
