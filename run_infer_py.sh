# 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES=3
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH
python3 infer.py --model_path=../weight/CosyVoice2-0.5B --stream
