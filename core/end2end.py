from core.asr import WhisperASR, AzureASR
from core.translator import LLMTranslator
from core.tts import EdgeTTS
from core.conversion import OpenVoiceConverter
import asyncio
import time
import azure.cognitiveservices.speech as speechsdk
import torchaudio
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

class End2End:
    def __init__(self):
        self.asr_model = AzureASR()
        self.translate_model = LLMTranslator()
        self.tts_model = EdgeTTS()
        self.converter = OpenVoiceConverter()


    def end2end_flow(self, source_language, target_language, audio):
        transcription = self.asr_model.transcribe_flow(audio, source_language)
        translate_text = self.translate_model.translate_flow(source_language, target_language, transcription)
        temp_file = asyncio.run(self.tts_model.tts_flow(target_language, translate_text))
        start = time.perf_counter()
        output_file = self.converter.convert(temp_file, audio)
        end = time.perf_counter()
        print(f'Conversion time: {end - start}')
        return output_file

class AzureEnd2End:
    def __init__(self):
        self.speech_key, self.service_region = os.getenv('AZURE_SPEECH_KEY'), os.getenv('AZURE_SERVICE_REGION')
        self.lang_mapping = {
            'zh': 'zh-TW',
            'en': 'en-US',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
        }
        self.lang2voice = {
            'zh' : 'zh-TW-HsiaoChenNeural',
            'en' : 'en-US-AvaNeural', 
            'ja' : 'ja-JP-KeitaNeural', 
            'ko' : 'ko-KR-HyunsuNeural'
        }
        self.converter = OpenVoiceConverter()

    def convert_16k(self, wav_file):
        data, sr = torchaudio.load(wav_file)
        if sr != 16000:
            data = torchaudio.functional.resample(data, sr, 16000)
            new_wav_file = wav_file.replace(".wav", "_16k.wav")
            torchaudio.save(new_wav_file, data, 16000)
            return new_wav_file
        return wav_file

    def callback_with_params(self, temp_file):
        def synthesis_callback(evt):
            size = len(evt.result.audio)
            print(f'Audio synthesized: {size} byte(s) {"(COMPLETED)" if size == 0 else ""}')

            if size > 0:
                file = open(temp_file, 'wb+')
                file.write(evt.result.audio)
                file.close()

        return synthesis_callback

    def get_result_text(self, reason, result):
        source_text = None
        target_text = None
        if reason == speechsdk.ResultReason.TranslatedSpeech:
            source_text = result.text
            target_text = result.translations[self.target_language]
            print(f'Recognized "{self.source_language}": {source_text}\n' + f'Translated into "{self.target_language}"": {target_text}')
        elif reason == speechsdk.ResultReason.RecognizedSpeech:
            source_text = result.text
            print(f'Recognized "{self.source_language}": {source_text}')
        elif reason == speechsdk.ResultReason.NoMatch:
            print(f'No speech could be recognized: {result.no_match_details}')
        elif reason == speechsdk.ResultReason.Canceled:
            print(f'Speech Recognition canceled: {result.cancellation_details}')

        return source_text, target_text

    def end2end_flow(self, source_language, target_language, audio):    
        
        self.source_language = source_language
        self.target_language = target_language
        audio = self.convert_16k(audio)
        speech_translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=self.speech_key, region=self.service_region)
        speech_translation_config.speech_recognition_language=self.lang_mapping[source_language]
        speech_translation_config.add_target_language(target_language)

        audio_config = speechsdk.audio.AudioConfig(filename=audio)

        speech_translation_config.voice_name = self.lang2voice[target_language]
        translation_recognizer = speechsdk.translation.TranslationRecognizer(translation_config=speech_translation_config, audio_config=audio_config)

        temp_file = audio.replace('_16k', '').replace('.wav', '_azure_temp.wav')
        translation_recognizer.synthesizing.connect(self.callback_with_params(temp_file))
        
        # self.temp_file = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/azure_temp.wav'
        start = time.perf_counter() 
        result = translation_recognizer.recognize_once()
        end = time.perf_counter()
        print(f'translation time: {end - start}')
        print(result)
        source_text, target_text = self.get_result_text(reason=result.reason, result=result)

        output_file = None
        if result.reason == speechsdk.ResultReason.TranslatedSpeech:
            start = time.perf_counter()
            output_file = self.converter.convert(temp_file, audio)
            end = time.perf_counter()
            print(f'Conversion time: {end - start}')

        if source_text is None:
            source_text = 'No speech could be recognized'
        if target_text is None:
            target_text = 'No text could be translated'

        return source_text, target_text, output_file