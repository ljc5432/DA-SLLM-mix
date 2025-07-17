from pyannote.audio import Pipeline
import torch

def get_diarization_segments(audio_path: str) -> list:
    # TODO: 在此实现pyannote.audio的调用逻辑
    # 输入: 音频路径
    # 输出: [{'speaker': 'SPK_01', 'start': 1.2, 'end': 3.4}, ...]
    # 初始化预训练管道
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="HF_TOKEN")  #  <------ hugging_face的token
    pipeline.to(torch.device("cuda"))
    # 执行说话人分离
    diarization = pipeline(audio_path)

    # 提取结果段落
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })

    return segments