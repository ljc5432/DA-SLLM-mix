import torch
import torchaudio
from pyannote.audio import Pipeline
from typing import List, Dict, Optional

# --- 全局模型加载 ---
# 将Diarization pipeline作为全局变量加载，以提高效率。
# 创建一个正确的 torch.device 对象
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE_ID = "pyannote/speaker-diarization-3.1"

try:
    print(f"正在加载 Diarization pipeline (模型: {PIPELINE_ID})...")
    # 如果需要，请传入你的Hugging Face token
    # from huggingface_hub import login
    # login("hf_...")
    DIARIZATION_PIPELINE = Pipeline.from_pretrained(PIPELINE_ID)
    DIARIZATION_PIPELINE.to(DEVICE)
    print("Diarization pipeline 加载成功。")
except Exception as e:
    print(f"加载 Diarization pipeline 失败: {e}")
    DIARIZATION_PIPELINE = None

def _resample_and_to_mono(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """
    将音频重采样到16kHz并转换为单声道。
    """
    target_sr = 16000
    # 重采样
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform

def get_diarization_segments(
    audio_path: str, 
    min_duration_s: float = 0.5,
    num_speakers: Optional[int] = None
) -> List[Dict]:
    """
    使用 pyannote.audio 对音频文件进行说话人日志分析。

    Args:
        audio_path (str): 音频文件的路径。
        min_duration_s (float): 要保留的语音片段的最小持续时间（秒）。
        num_speakers (Optional[int]): 如果已知说话人数量，可以传入以提高准确率。

    Returns:
        List[Dict]: 一个包含所有语音片段信息的列表，
                    每个元素为 {'speaker': str, 'start': float, 'end': float}。
    """
    if DIARIZATION_PIPELINE is None:
        raise RuntimeError("Diarization pipeline未能成功加载，请检查模型ID、网络连接或用户协议。")

    print(f"开始对 {audio_path} 进行说话人日志分析...")

    try:
        # 1. 加载并预处理音频
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = _resample_and_to_mono(waveform, sample_rate)
        
        # 2. 运行Diarization流水线
        # 可以传入已知说话人数量来优化结果
        diarization_params = {}
        if num_speakers is not None and num_speakers > 0:
            diarization_params["num_speakers"] = num_speakers
            print(f"使用已知的说话人数量: {num_speakers}")

        diarization_result = DIARIZATION_PIPELINE(
            {"waveform": waveform, "sample_rate": 16000},
            **diarization_params
        )

        # 3. 格式化输出
        segments = []
        for segment, _, speaker_id in diarization_result.itertracks(yield_label=True):
            duration = segment.end - segment.start
            if duration >= min_duration_s:
                segments.append({
                    'speaker': speaker_id,
                    'start': round(segment.start, 3),
                    'end': round(segment.end, 3)
                })
        
        # pyannote的输出已经是按时间排序的，但以防万一再次排序
        segments.sort(key=lambda x: x['start'])
        
        print(f"说话人日志分析完成，共找到 {len(segments)} 个有效语音片段。")
        return segments

    except Exception as e:
        print(f"错误：在Diarization过程中失败: {e}")
        return []