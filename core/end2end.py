from core.asr import WhisperASR, AzureASR
from core.translator import LLMTranslator
from core.tts import EdgeTTS
from core.conversion import OpenVoiceConverter, RVCConverter
import asyncio
import time
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError
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
        self.speech_key, self.service_region, self.speech_endpoint = os.getenv('AZURE_CUSTOM_SPEECH_KEY'), os.getenv('AZURE_CUSTOM_SERVICE_REGION'), os.getenv('AZURE_CUSTOM_ENDPOINT')
        self.translation_key, self.translation_endpoint, self.translation_region = os.getenv('AZURE_TRANSLATION_KEY'), os.getenv('AZURE_TRANSLATION_ENDPOINT'), os.getenv('AZURE_TRANSLATION_REGION')
        self.lang_mapping = {
            'zh': 'zh-TW',
            'en': 'en-US',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'pl': 'pl-PL',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'nl': 'nl-NL',
        }
        self.lang2voice = {
            # language code : [female voice name, male voice name]
            'zh' : ['zh-TW-HsiaoChenNeural', 'zh-TW-YunJheNeural'],
            'en' : ['en-US-AvaNeural', 'en-US-AndrewNeural'], 
            'ja' : ['ja-JP-NanamiNeural', 'ja-JP-KeitaNeural'], 
            'ko' : ['ko-KR-SunHiNeural', 'ko-KR-InJoonNeural'], 
            'pl' : ['pl-PL-AgnieszkaNeural', 'pl-PL-MarekNeural'], 
            'fr' : ['fr-FR-DeniseNeural', 'fr-FR-HenriNeural'], 
            'de' : ['de-DE-KatjaNeural', 'de-DE-ConradNeural'],
            'nl' : ['nl-NL-FennaNeural', 'nl-NL-MaartenNeural']
        }
        self.converter = OpenVoiceConverter()
        self.custom_converter = RVCConverter()

    def get_support_languages(self):
        return list(self.lang_mapping.keys())
    
    def get_custom_models(self):
        return ['auto'] + list(self.custom_converter.customs_model_with_name.keys())

    def convert_16k(self, wav_file):
        data, sr = torchaudio.load(wav_file)
        if sr != 16000:
            data = torchaudio.functional.resample(data, sr, 16000)
            new_wav_file = wav_file.replace(".wav", "_16k.wav")
            torchaudio.save(new_wav_file, data, 16000)
            return new_wav_file
        return wav_file
    
    def speech_translation_continous(self, audio, temp_file, vc_model_name):
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

        if self.source_language == 'zh':   
            # use custom zh-TW model
            print('use custom zh-TW model')
            speech_translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=os.getenv('AZURE_CUSTOM_SPEECH_KEY'), region=os.getenv('AZURE_CUSTOM_SERVICE_REGION'))
            speech_translation_config.endpoint_id=os.getenv('AZURE_CUSTOM_ENDPOINT')
        else:
            print('use default model')
            speech_translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=os.getenv('AZURE_SPEECH_KEY'), region=os.getenv('AZURE_SERVICE_REGION'))
        speech_translation_config.speech_recognition_language=self.lang_mapping[self.source_language]
        speech_translation_config.add_target_language(self.target_language if self.target_language != 'zh' else 'zh-Hant')
        speech_translation_config.voice_name = self.lang2voice[self.target_language][0] if vc_model_name != 'chris' else self.lang2voice[self.target_language][1]
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
    
    def speech_translation_continous_src_langdetect(self, audio, temp_file, vc_model_name):
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
                detect_src_lang = evt.result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
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

        endpoint_string = "wss://{}.stt.speech.microsoft.com/speech/universal/v2".format(os.getenv('AZURE_SERVICE_REGION'))
        speech_translation_config = speechsdk.translation.SpeechTranslationConfig(
            subscription=os.getenv('AZURE_SPEECH_KEY'),
            endpoint=endpoint_string,)
        speech_translation_config.add_target_language(self.target_language if self.target_language != 'zh' else 'zh-Hant')
        speech_translation_config.voice_name = self.lang2voice[self.target_language][0] if vc_model_name != 'chris' else self.lang2voice[self.target_language][1]
        audio_config = speechsdk.audio.AudioConfig(filename=audio)
        # Since the spoken language in the input audio changes, you need to set the language identification to "Continuous" mode.
        # (override the default value of "AtStart").
        speech_translation_config.set_property(
            property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value='Continuous')
        auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=list(self.lang_mapping.values()))
        
        translation_recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=speech_translation_config, 
            audio_config=audio_config, 
            auto_detect_source_language_config=auto_detect_source_language_config)

        # connect callback functions to the events fired by the recognizer
        # translation_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
        # translation_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
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

    def sts_end2end_pipeline(self, source_language, target_language, audio, vc_model_name=None):    
        
        self.source_language = source_language
        self.target_language = target_language
        audio = self.convert_16k(audio)
        temp_file = audio.replace('_16k', '').replace('.wav', '_azure_temp.wav')

        start = time.perf_counter() 
        source_text, target_text, asr_timestamps = self.speech_translation_continous_src_langdetect(audio, temp_file, vc_model_name)
        end = time.perf_counter()
        print(f'translation time: {end - start}')
        output_file = None
        if source_text != '' and target_text != '':
            start = time.perf_counter()
            if vc_model_name == 'auto':
                output_file = self.converter.convert(temp_file, audio, source_timestamps='all', target_timestamps=asr_timestamps)
            else:
                output_file = self.custom_converter.convert(temp_file, model_name=vc_model_name)
            end = time.perf_counter()
            print(f'Conversion time: {end - start}')

        if source_text == '':
            source_text = 'No speech could be recognized'
        if target_text == '':
            target_text = 'No text could be translated'

        return source_text, target_text, output_file
    
    def text_translation(self, text):
        credential = TranslatorCredential(self.translation_key, self.translation_region)
        text_translator = TextTranslationClient(endpoint=self.translation_endpoint, credential=credential)

        target_text = ''
        
        try:
            input_text_elements = [ InputTextItem(text = text) ]

            response = text_translator.translate(content = input_text_elements, 
                                                 to = [self.target_language if self.target_language != 'zh' else 'zh-Hant'], 
                                                 from_parameter = self.source_language if self.source_language != 'zh' else 'zh-Hant')
            translation = response[0] if response else None

            if translation:
                print(f"Text was translated to: '{translation.translations[0].to}' and the result is: '{translation.translations[0].text}'.")
                target_text = translation.translations[0].text

        except HttpResponseError as exception:
            print(f"Error Code: {exception.error.code}")
            print(f"Message: {exception.error.message}")

        return target_text
    
    def text_to_speech(self, target_text, temp_file, vc_model_name):
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_SPEECH_KEY'), region=os.getenv('AZURE_SERVICE_REGION'))
        speech_config.speech_synthesis_voice_name = self.lang2voice[self.target_language][0] if vc_model_name != 'chris' else self.lang2voice[self.target_language][1]
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_file)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_synthesizer.speak_text_async(target_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}], and the audio was saved to [{}]".format(target_text, temp_file))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

    def tts_end2end_pipeline(self, source_language, target_language, source_text, audio, vc_model_name=None):

        self.source_language = source_language
        self.target_language = target_language
        temp_file = audio.replace('_16k', '').replace('.wav', '_azure_temp.wav')

        start = time.perf_counter() 
        target_text = self.text_translation(source_text)
        end = time.perf_counter()
        print(f'translation time: {end - start}')

        output_file = None
        if source_text != '' and target_text != '':
            start = time.perf_counter()
            self.text_to_speech(target_text, temp_file, vc_model_name)
            end = time.perf_counter()
            print(f'Text to Speech time: {end - start}')

            start = time.perf_counter()
            if vc_model_name == 'auto':
                print('didn\'t give vc_model_name, return tts wav file directly')
                output_file = temp_file
            else:
                output_file = self.custom_converter.convert(temp_file, model_name=vc_model_name)
            
            end = time.perf_counter()
            print(f'Conversion time: {end - start}')

        if source_text == '':
            source_text = 'No speech could be recognized'
        if target_text == '':
            target_text = 'No text could be translated'

        return source_text, target_text, output_file