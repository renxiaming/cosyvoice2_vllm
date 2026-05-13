#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# mypy: ignore-errors
from functools import cache
from typing import Optional

import torch
import vllm
import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.selector import (
    backend_name_to_enum,
    get_global_forced_attn_backend,
)
from vllm.platforms import _Backend, current_platform
from vllm.utils import resolve_obj_by_qualname

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.10.2"):

    def get_attn_backend(
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        is_attention_free: bool = False,
        use_mla: bool = False,
        use_sfa: bool = False,
        has_sink: bool = False,
        use_sparse: bool = False,
    ) -> type[AttentionBackend]:
        """0.10.2 路径；多出的 use_sparse 仅兼容，不参与 0.10.2 平台选择。"""
        return _cached_get_attn_backend(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            is_attention_free=is_attention_free,
            use_v1=envs.VLLM_USE_V1,
            use_mla=use_mla,
            use_sfa=use_sfa,
            has_sink=has_sink,
            use_sparse=use_sparse,
        )

    @cache
    def _cached_get_attn_backend(
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        is_attention_free: bool,
        use_v1: bool = False,
        use_mla: bool = False,
        use_sfa: bool = False,
        has_sink: bool = False,
        use_sparse: bool = False,
    ) -> type[AttentionBackend]:
        _ = use_sparse

        if is_attention_free:
            from vllm.attention.backends.placeholder_attn import (
                PlaceholderAttentionBackend,
            )

            return PlaceholderAttentionBackend

        selected_backend = None
        backend_by_global_setting: Optional[_Backend] = (
            get_global_forced_attn_backend())
        if backend_by_global_setting is not None:
            selected_backend = backend_by_global_setting
        else:
            backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
            if backend_by_env_var is not None:
                selected_backend = backend_name_to_enum(backend_by_env_var)
                if selected_backend is None:
                    raise ValueError(
                        f"Invalid attention backend: '{backend_by_env_var}'. "
                        f"Valid backends are: {list(_Backend.__members__.keys())}"
                    )

        attention_cls = current_platform.get_attn_backend_cls(
            selected_backend, head_size, dtype, kv_cache_dtype, block_size,
            use_v1, use_mla, use_sfa, has_sink)
        if not attention_cls:
            raise ValueError(
                f"Invalid attention backend for {current_platform.device_name}"
            )
        return resolve_obj_by_qualname(attention_cls)
else:

    def get_attn_backend(  # type: ignore[misc]
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_mla: bool = False,
        has_sink: bool = False,
        use_sparse: bool = False,
        use_sfa: bool = False,
    ) -> type[AttentionBackend]:
        """对齐 vLLM 0.11 selector；兼容 ascend 内部 ``use_sfa=...`` 调用。"""
        _ = use_sfa
        return _cached_get_attn_backend(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_v1=envs.VLLM_USE_V1,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
        )

    @cache
    def _cached_get_attn_backend(
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool = False,
        use_mla: bool = False,
        has_sink: bool = False,
        use_sparse: bool = False,
    ) -> type[AttentionBackend]:
        selected_backend = None
        backend_by_global_setting: Optional[_Backend] = (
            get_global_forced_attn_backend())
        if backend_by_global_setting is not None:
            selected_backend = backend_by_global_setting
        else:
            backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
            if backend_by_env_var is not None:
                selected_backend = backend_name_to_enum(backend_by_env_var)
                if selected_backend is None:
                    raise ValueError(
                        f"Invalid attention backend: '{backend_by_env_var}'. "
                        f"Valid backends are: {list(_Backend.__members__.keys())}"
                    )

        # 与 vLLM 0.11 ``selector._cached_get_attn_backend`` 一致
        attention_cls = current_platform.get_attn_backend_cls(
            selected_backend, head_size, dtype, kv_cache_dtype, block_size,
            use_v1, use_mla, has_sink, use_sparse)
        if not attention_cls:
            raise ValueError(
                f"Invalid attention backend for {current_platform.device_name}"
            )
        return resolve_obj_by_qualname(attention_cls)


import vllm.attention.selector as _vllm_attn_selector

_vllm_attn_selector.get_attn_backend = get_attn_backend
_vllm_attn_selector._cached_get_attn_backend = _cached_get_attn_backend
vllm.attention.get_attn_backend = get_attn_backend