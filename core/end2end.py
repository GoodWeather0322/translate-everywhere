from core.asr import WhisperASR, AzureASR
from core.translator import LLMTranslator
from core.tts import EdgeTTS
from core.conversion import OpenVoiceConverter, RVCConverter
import asyncio
import time
import azure.cognitiveservices.speech as speechsdk
import torchaudio
from dotenv import load_dotenv
import os
from pydub import AudioSegment

# Load environment variables from a .env file
load_dotenv()

class End2End:
    def __init__(self):
        self.asr_model = AzureASR()
        self.translate_model = LLMTranslator()
        self.tts_model = EdgeTTS()
        self.converter = OpenVoiceConverter()

    def end2end_pipeline(self, source_language, target_language, audio):
        transcription = self.asr_model.transcribe_pipeline(audio, source_language)
        translate_text = self.translate_model.translate_pipeline(source_language, target_language, transcription)
        temp_file = asyncio.run(self.tts_model.tts_pipeline(target_language, translate_text))
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
            'pl': 'pl-PL',
            'fr': 'fr-FR',
        }
        self.lang2voice = {
            # language code : [female voice name, male voice name]
            'zh' : ['zh-TW-HsiaoChenNeural', 'zh-TW-YunJheNeural'],
            'en' : ['en-US-AvaNeural', 'en-US-AndrewNeural'], 
            'ja' : ['ja-JP-NanamiNeural', 'ja-JP-KeitaNeural'], 
            'ko' : ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural'], 
            'pl' : ['pl-PL-AgnieszkaNeural', 'pl-PL-MarekNeural'], 
            'fr' : ['fr-FR-DeniseNeural', 'fr-FR-HenriNeural'], 
        }
        self.converter = OpenVoiceConverter()
        self.custom_converter = RVCConverter()

    def get_support_languages(self):
        return list(self.lang_mapping.keys())
    
    def get_custom_models(self):
        return list(self.custom_converter.customs_model_with_name.keys())

    def convert_16k(self, wav_file):
        data, sr = torchaudio.load(wav_file)
        if sr != 16000:
            data = torchaudio.functional.resample(data, sr, 16000)
            new_wav_file = wav_file.replace(".wav", "_16k.wav")
            torchaudio.save(new_wav_file, data, 16000)
            return new_wav_file
        return wav_file
    
    def translation_continous(self, audio, temp_file):
        """performs continuous speech translation from an audio file"""

        done = False
        source_text = []
        target_text = []
        temp_synthesis_files = []
        asr_timestamps = []

        def result_callback(event_type: str, evt: speechsdk.translation.TranslationRecognitionEventArgs):
            """callback to display a translation result"""
            nonlocal source_text, target_text, asr_timestamps
            if event_type == 'RECOGNIZED':
                source_text.append(evt.result.text)
                target_text.append(evt.result.translations[self.target_language if self.target_language != 'zh' else 'zh-Hant'])
                offset = evt.result.offset
                duration = evt.result.duration
                start = (offset / 10000000)
                end = start + (duration / 10000000)
                asr_timestamps.append((start, end))
                print("{}:\tTranslations: {}\n\tResult Json: {}".format(
                    event_type, evt.result.translations.items(), evt.result.json))
            
        def stop_cb(evt: speechsdk.SessionEventArgs):
            """callback that signals to stop continuous recognition upon receiving an event `evt`"""
            # print('CLOSING on {}'.format(evt))
            nonlocal done
            done = True

        def canceled_cb(evt: speechsdk.translation.TranslationRecognitionCanceledEventArgs):
            pass
            # print('CANCELED:\n\tReason:{}\n'.format(evt.result.reason))
            # print('\tDetails: {} ({})'.format(evt, evt.result.cancellation_details.error_details))

        def synthesis_callback(evt: speechsdk.translation.TranslationRecognitionEventArgs):
            """
            callback for the synthesis event
            """
            nonlocal temp_synthesis_files
            synthesis_bytes = evt.result.audio
            if len(synthesis_bytes) > 0:
                temp_synthesis_file = temp_file.replace('.wav', f'_{len(temp_synthesis_files)}.wav')
                with open(temp_synthesis_file, 'wb+') as f:
                    f.write(synthesis_bytes)
                temp_synthesis_files.append(temp_synthesis_file)
            print('SYNTHESIZING: \treceived {} bytes of audio. Reason: {}'.format(
                len(evt.result.audio), evt.result.reason))
            
        def save_synthesis(temp_synthesis_files, temp_file):
            combined_audio = AudioSegment.empty()
            for temp_synthesis_file in temp_synthesis_files:
                audio = AudioSegment.from_file(temp_synthesis_file)
                combined_audio += audio

            combined_audio.export(temp_file, format="wav")
            for temp_synthesis_file in temp_synthesis_files:
                os.remove(temp_synthesis_file)
                
        speech_translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=self.speech_key, region=self.service_region)
        speech_translation_config.speech_recognition_language=self.lang_mapping[self.source_language]
        speech_translation_config.add_target_language(self.target_language if self.target_language != 'zh' else 'zh-Hant')
        speech_translation_config.voice_name = self.lang2voice[self.target_language][0]
        audio_config = speechsdk.audio.AudioConfig(filename=audio)
        translation_recognizer = speechsdk.translation.TranslationRecognizer(translation_config=speech_translation_config, audio_config=audio_config)

        # connect callback functions to the events fired by the recognizer
        # translation_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        # translation_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
        # event for intermediate results
        translation_recognizer.recognizing.connect(lambda evt: result_callback('RECOGNIZING', evt))
        # event for final result
        translation_recognizer.recognized.connect(lambda evt: result_callback('RECOGNIZED', evt))
        # cancellation event
        translation_recognizer.canceled.connect(canceled_cb)

        # stop continuous recognition on either session stopped or canceled events
        translation_recognizer.session_stopped.connect(stop_cb)
        translation_recognizer.canceled.connect(stop_cb)
        # connect callback to the synthesis event
        translation_recognizer.synthesizing.connect(synthesis_callback)

        translation_recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.1)
        translation_recognizer.stop_continuous_recognition()
        if len(temp_synthesis_files) > 0:
            save_synthesis(temp_synthesis_files, temp_file)
        return ' '.join(source_text), ' '.join(target_text), asr_timestamps
        

    def end2end_pipeline(self, source_language, target_language, audio, vc_model_name=None):    
        
        self.source_language = source_language
        self.target_language = target_language
        audio = self.convert_16k(audio)
        temp_file = audio.replace('_16k', '').replace('.wav', '_azure_temp.wav')

        start = time.perf_counter() 
        source_text, target_text, asr_timestamps = self.translation_continous(audio, temp_file)
        end = time.perf_counter()
        print(f'translation time: {end - start}')
        output_file = None
        if source_text != '' and target_text != '':
            start = time.perf_counter()
            if vc_model_name:
                output_file = self.custom_converter.convert(temp_file, model_name=vc_model_name)
            else:
                output_file = self.converter.convert(temp_file, audio, source_timestamps='all', target_timestamps=asr_timestamps)
            end = time.perf_counter()
            print(f'Conversion time: {end - start}')

        if source_text == '':
            source_text = 'No speech could be recognized'
        if target_text == '':
            target_text = 'No text could be translated'

        return source_text, target_text, output_file