# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 模仿n卡vllm推理导入
import sys
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
import os

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
            for i, j in enumerate(cosyvoice.inference_sft(prompt_txt, '中文女', stream=args.stream)):
                torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)