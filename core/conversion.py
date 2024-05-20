import sys
sys.path.append("/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/OpenVoice")
import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import time
from pydub import AudioSegment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

class ConverterBase:
    def __init__(self):
        pass    

    def convert(self, source_wav, target_wav):
        raise NotImplementedError
    
class OpenVoiceConverter(ConverterBase):
    def __init__(self):
        super().__init__()
        ckpt_converter = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/OpenVoice/checkpoints_v2/converter'
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        
        self.vad_model = Model.from_pretrained(
                "pyannote/segmentation-3.0", 
                use_auth_token="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")
            
        self.pipeline = VoiceActivityDetection(segmentation=self.vad_model)
        HYPER_PARAMETERS = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.0
        }
        self.pipeline.instantiate(HYPER_PARAMETERS)
    
        print(f'Load tone color converter from {ckpt_converter}')

    # def split_audio_whisper(self, audio_path, target_dir='processed', audio_name=None):
    #     audio = AudioSegment.from_file(audio_path)
    #     max_len = len(audio)

    #     target_folder = os.path.join(target_dir, audio_name)
        
    #     segments, info = self.whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    #     segments = list(segments)    

    #     # create directory
    #     os.makedirs(target_folder, exist_ok=True)
    #     wavs_folder = os.path.join(target_folder, 'wavs')
    #     os.makedirs(wavs_folder, exist_ok=True)

    #     # segments
    #     s_ind = 0
    #     start_time = None
        
    #     for k, w in enumerate(segments):
    #         # process with the time
    #         if k == 0:
    #             start_time = max(0, w.start)

    #         end_time = w.end

    #         # calculate confidence
    #         if len(w.words) > 0:
    #             confidence = sum([s.probability for s in w.words]) / len(w.words)
    #         else:
    #             confidence = 0.
    #         # clean text
    #         text = w.text.replace('...', '')

    #         # left 0.08s for each audios
    #         audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

    #         # segment file name
    #         fname = f"{audio_name}_seg{s_ind}.wav"

    #         # filter out the segment shorter than 1.5s and longer than 20s
    #         save = audio_seg.duration_seconds > 1.5 and \
    #                 audio_seg.duration_seconds < 20. and \
    #                 len(text) >= 2 and len(text) < 200 

    #         if save:
    #             output_file = os.path.join(wavs_folder, fname)
    #             audio_seg.export(output_file, format='wav')

    #         if k < len(segments) - 1:
    #             start_time = max(0, segments[k+1].start - 0.08)

    #         s_ind = s_ind + 1
    #     return wavs_folder

    def split_audio_vad(self, audio_path, target_dir='processed', audio_name=None):
        audio = AudioSegment.from_file(audio_path)
        max_len = len(audio)
        target_folder = os.path.join(target_dir, audio_name)

        
        vad = self.pipeline(audio_path)
        segments = list(vad.get_timeline().support())    

        # create directory
        os.makedirs(target_folder, exist_ok=True)
        wavs_folder = os.path.join(target_folder, 'wavs')
        os.makedirs(wavs_folder, exist_ok=True)

        # segments
        s_ind = 0
        start_time = None
        
        for k, w in enumerate(segments):
            # process with the time
            if k == 0:
                start_time = max(0, w.start)

            end_time = w.end

            # left 0.08s for each audios
            audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

            # segment file name
            fname = f"{audio_name}_seg{s_ind}.wav"

            # filter out the segment shorter than 1.5s and longer than 20s
            save = audio_seg.duration_seconds > 1.5 and \
                    audio_seg.duration_seconds < 20.

            if save:
                output_file = os.path.join(wavs_folder, fname)
                audio_seg.export(output_file, format='wav')

            if k < len(segments) - 1:
                start_time = max(0, segments[k+1].start - 0.08)

            s_ind = s_ind + 1
        return wavs_folder

    def get_se(self, audio_path, vc_model, target_dir='processed', vad=True):
        device = vc_model.device
        version = vc_model.version
        print("OpenVoice version:", version)

        audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{se_extractor.hash_numpy_array(audio_path)}"
        se_path = os.path.join(target_dir, audio_name, 'se.pth')

        # if os.path.isfile(se_path):
        #     se = torch.load(se_path).to(device)
        #     return se, audio_name
        # if os.path.isdir(audio_path):
        #     wavs_folder = audio_path

        if vad:
            wavs_folder = self.split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
        else:
            wavs_folder = self.split_audio_whisper(audio_path, target_dir=target_dir, audio_name=audio_name)

        from glob import glob
        audio_segs = glob(f'{wavs_folder}/*.wav')
        if len(audio_segs) == 0:
            raise NotImplementedError('No audio segments found!')
        
        return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name

    def convert(self, source_wav, target_wav):
        start = time.perf_counter()
        source_se, source_audio_name = self.get_se(source_wav, self.tone_color_converter, vad=True)
        target_se, target_audio_name = self.get_se(target_wav, self.tone_color_converter, vad=True)
        end = time.perf_counter()
        print(f'Get source and target se: {end - start:.4f}s')

        save_path = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/final_conversion.wav'

        start = time.perf_counter()
        encode_message = "@translate_everywhere"
        self.tone_color_converter.convert(
            audio_src_path=source_wav, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        end = time.perf_counter()
        print(f'Convert: {end - start:.4f}s')
        
        return save_path