# 指定使用NPU ID，默认为0（910B 系列）
export ASCEND_RT_VISIBLE_DEVICES=0
# 不要在仓库根目录保留名为 vllm/ 的源码树：会先于 site-packages 被 import，
# 导致 vllm_ascend 里 `from vllm import ModelRegistry` 失败。infer_vllm.py 会尝试纠正。
# vLLM 0.11 默认 V1 引擎；Ascend 由 vllm-ascend 插件注册（日志里 ascend -> vllm_ascend:register）
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# EngineCore 子进程：infer_vllm.py 已将 multiprocessing 设为 spawn（NPU 不可在 fork 子进程里再 init）
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH

# NOTE:
# This repo vendors a `transformers/src` checkout, but it may be older than what
# vLLM 0.11.x expects (e.g. missing `AutoVideoProcessor`). By default, prefer
# the environment-installed `transformers` package. If you must use the vendored
# copy, set `USE_LOCAL_TRANSFORMERS=1`.
if [ "${USE_LOCAL_TRANSFORMERS:-0}" = "1" ]; then
  export PYTHONPATH=transformers/src:$PYTHONPATH
fi

# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 规避找不到ttsfrd
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 规避找不到cstdint
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH}

# 清理modelscope缓存
rm -rf ~/.cache/modelscope/

python3 infer_vllm.py --model_path=../weight/CosyVoice2-0.5B --stream
