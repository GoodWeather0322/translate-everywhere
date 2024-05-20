from faster_whisper import WhisperModel
import azure.cognitiveservices.speech as speechsdk
import numpy as np
import torchaudio
from torchaudio.transforms import Resample
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

class ASRBase:
    def __init__(self):
        self.sample_rate = 16000

    def recv_audio_offline(self, audio):
        waveform, original_sample_rate = torchaudio.load(audio)
        if original_sample_rate != self.sample_rate:
            resampler = Resample(original_sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy()
    
    def transcribe(self, audio):
        raise NotImplementedError
    
    def transcribe_flow(self, audio):
        raise NotImplementedError
    

class WhisperASR(ASRBase):
    def __init__(self):
        super().__init__()
        model_size = "large-v3"
        # # Run on GPU with FP16
        # # model = WhisperModel(model_size, device="cuda", compute_type="float16")

        # # or run on GPU with INT8
        self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # # or run on CPU with INT8
        # # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def recv_audio_offline(self, audio):
        return super().recv_audio_offline(audio)

    def transcribe_flow(self, audio, language):

        if not isinstance(audio, np.ndarray):
            audio = self.recv_audio_offline(audio)

        segments, info = self.model.transcribe(audio, beam_size=5, language=language, task='transcribe')
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        final_text = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            final_text.append(segment.text)

        final_text = ' '.join(final_text)

        return final_text

class AzureASR(ASRBase):
    def __init__(self):
        pass
    
    def transcribe_flow(self, audio, language):

        def convert_16k(wav_file):
            data, sr = torchaudio.load(wav_file)
            if sr != 16000:
                data = torchaudio.functional.resample(data, sr, 16000)
                new_wav_file = wav_file.replace(".wav", "_16k.wav")
                torchaudio.save(new_wav_file, data, 16000)
                return new_wav_file
            return wav_file
        def covert_lang(lang):
            mapping = {
                "en": "en-US",
                "zh": "zh-TW",
                "ja": "ja-JP",
                "ko": "ko-KR",
            }
            if lang in mapping:
                return mapping[lang]
            else:
                return "en-US"
        
        audio = convert_16k(audio)

        speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_SPEECH_KEY'), region=os.getenv('AZURE_SERVICE_REGION'))
        speech_config.speech_recognition_language=covert_lang(language)
        audio_config = speechsdk.AudioConfig(filename=audio)

        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once()
        print(result.text)
        return result.text