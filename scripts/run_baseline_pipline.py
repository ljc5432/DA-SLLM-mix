# scripts/run_baseline_pipeline.py (示例)

import os
import sys
import pprint # 用于美观地打印结果

# --- 设置环境 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- 导入我们自己的模块 ---
# 假设 diarization.py 已经实现
from diarization_aware_asr.baseline import diarization, asr 

def main():
    # 使用您提供的音频文件路径
    audio_file = "/media/zjnu/c0369a81-a791-4056-a70f-aa9730441592/home/zjnu/voice_LLM/ljc/DA-SLLM-mix/datasets/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav"
    
    if not os.path.exists(audio_file):
        print(f"错误：音频文件未找到于 {audio_file}")
        return

    print("--- 步骤 1: 说话人日志 (Diarization) ---")
    # 调用真实的Diarization函数
    # 我们可以尝试传入已知的说话人数量，对于AliMeeting的评估集，通常可以从文件名或元数据中得知
    # 如果不知道，就不传入 num_speakers 参数
    diarization_result = diarization.get_diarization_segments(audio_file, num_speakers=None)

    if not diarization_result:
        print("Diarization未能识别出任何语音片段，程序结束。")
        return
    
    print("\nDiarization 结果 (前5条):")
    pprint.pprint(diarization_result[:5])

    print("\n--- 步骤 2: 语音识别 (ASR) ---")
    transcribed_result = asr.transcribe_segments(audio_file, diarization_result)
    
    print("\n--- 最终转写结果 (前5条) ---")
    pprint.pprint(transcribed_result[:5])

    # TODO: 在这里可以添加步骤3，调用 formatter.py 将 transcribed_result 转换成Markdown文本

if __name__ == "__main__":
    main()