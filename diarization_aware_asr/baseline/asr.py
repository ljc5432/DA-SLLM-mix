# diarization_aware_asr/baseline/asr.py

import torch
import torchaudio
from transformers import pipeline
from typing import List, Dict
from tqdm import tqdm
import numpy as np

# --- 全局模型加载 ---
# 将模型和pipeline作为全局变量加载，避免在每次函数调用时都重新加载模型。
# 这在实际应用中非常重要，可以节省大量时间。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/whisper-large-v3"

# 使用 pipeline API，这是最高效的方式
# torch_dtype=torch.float16 和 device_map="auto" 可以优化显存使用和速度
# 如果显存不足，可以考虑 load_in_8bit=True 或 load_in_4bit=True
try:
    print(f"正在加载 Whisper ASR pipeline (模型: {MODEL_ID})...")
    ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto", # 自动将模型分到多个GPU（如果可用）
    )
    print("ASR pipeline 加载成功。")
except Exception as e:
    print(f"加载ASR pipeline失败: {e}")
    ASR_PIPELINE = None

def transcribe_segments(audio_path: str, segments: List[Dict], batch_size: int = 16) -> List[Dict]:
    """
    使用 whisper-large-v3 对Diarization后的音频片段进行批量转写。

    Args:
        audio_path (str): 完整音频文件的路径。
        segments (List[Dict]): Diarization结果的列表，每个元素应包含 'start' 和 'end' 键（单位：秒）。
        batch_size (int): 一次送入模型处理的音频片段数量。

    Returns:
        List[Dict]: 更新后的segments列表，每个元素都添加了 'text' 键。
    """
    if ASR_PIPELINE is None:
        raise RuntimeError("ASR pipeline未能成功加载，请检查模型ID、网络连接或依赖库。")
    
    if not segments:
        print("警告: 传入的segments列表为空，无需转写。")
        return []

    print(f"开始对 {len(segments)} 个音频片段进行批量转写 (batch_size={batch_size})...")

    try:
        # 1. 一次性加载完整音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        # 确保采样率为16kHz，这是Whisper需要的
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        # 确保是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 定义一个略小于30秒的阈值，以留出安全边际
        MAX_DURATION_S = 29.5
        
        # 2. 根据时间戳裁剪出所有音频片段的波形
        audio_clips = []

        # 这个列表将保存每个clip对应的原始segment索引，以便后续拼接文本
        clip_to_segment_map = [] 

        print("正在准备和切分音频片段...")
        for seg_idx, seg in enumerate(segments):
            seg_duration = seg['end'] - seg['start']
            
            if seg_duration <= MAX_DURATION_S:
                # 片段长度合适，直接裁剪
                start_frame = int(seg['start'] * 16000)
                end_frame = int(seg['end'] * 16000)
                clip = waveform[0, start_frame:end_frame].numpy()
                audio_clips.append(clip)
                clip_to_segment_map.append(seg_idx)
            else:
                # 片段太长，需要切分
                num_chunks = int(np.ceil(seg_duration / MAX_DURATION_S))
                chunk_duration_s = seg_duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start_s = seg['start'] + i * chunk_duration_s
                    chunk_end_s = chunk_start_s + chunk_duration_s
                    
                    start_frame = int(chunk_start_s * 16000)
                    end_frame = int(chunk_end_s * 16000)
                    
                    # 确保最后一个块的结束位置不超过原始片段的结束位置
                    if i == num_chunks - 1:
                        end_frame = int(seg['end'] * 16000)

                    clip = waveform[0, start_frame:end_frame].numpy()
                    audio_clips.append(clip)
                    clip_to_segment_map.append(seg_idx) # 多个小块都映射到同一个原始segment
            
    except Exception as e:
        print(f"错误：加载或裁剪音频文件 {audio_path} 时失败: {e}")
        return segments # 返回原始segments

    # 3. 使用pipeline进行批量转写
    # pipeline可以接受一个音频片段的列表
    # generate_kwargs可以指定语言，对于中英文混合的AliMeeting，不指定让其自动检测通常效果不错
    # 或者可以指定 language="chinese"
    try:
        transcriptions = []
        # 使用tqdm显示进度条
        for i in tqdm(range(0, len(audio_clips), batch_size), desc="Transcribing Batches"):
            batch_clips = audio_clips[i:i+batch_size]
            # `pipeline`返回一个字典列表，每个字典包含'text'键
            batch_results = ASR_PIPELINE(
                batch_clips, 
                generate_kwargs={"language": "chinese", "task": "transcribe"},
                return_timestamps=False # 我们已经有更准的Diarization时间戳，这里不需要Whisper的时间戳
            )
            # 提取文本
            texts = [res['text'] for res in batch_results]
            transcriptions.extend(texts)

    except Exception as e:
        print(f"错误：在ASR推理过程中失败: {e}")
        # 即使失败，也尝试返回已经转写的部分
        for i, text in enumerate(transcriptions):
            segments[i]['text'] = text.strip()
        return segments

    # 4. 将转写文本添加/拼接回原始的segments列表
    # 初始化所有原始segment的text字段
    for seg in segments:
        seg['text'] = ''
        
    # 遍历所有转写结果，根据映射关系将文本拼接回去
    for clip_idx, text in enumerate(transcriptions):
        original_seg_idx = clip_to_segment_map[clip_idx]
        # 使用空格连接被切分的片段的文本
        if segments[original_seg_idx]['text']:
            segments[original_seg_idx]['text'] += " " + text.strip()
        else:
            segments[original_seg_idx]['text'] = text.strip()

    print("转写完成。")
    return segments