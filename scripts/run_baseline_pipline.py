from diarization_aware_asr.baseline import diarization, asr, formatter
def main():
    audio_file = "path/to/your/test_audio.wav"
    # 1. Diarization
    diarization_result = diarization.get_diarization_segments(audio_file)
    # 2. ASR
    transcribed_result = asr.transcribe_segments(audio_file, diarization_result)
    # 3. Format
    # final_text = formatter.format_to_markdown(transcribed_result)
    # print(final_text)