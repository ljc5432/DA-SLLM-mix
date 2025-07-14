def transcribe_segments(audio_path: str, segments: list) -> list:
    # TODO: 在此实现whisper-large-v3的调用逻辑
    # 1. 根据segments中的时间戳，从audio_path中批量裁剪音频片段
    # 2. 将片段送入ASR模型进行批量转写
    # 3. 将转写文本添加回segments列表
    # 输入: 音频路径和Diarization结果
    # 输出: [{'speaker': 'SPK_01', 'start': 1.2, 'end': 3.4, 'text': '...'}, ...]
    pass
