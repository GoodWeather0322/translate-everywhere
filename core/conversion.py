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
import soundfile

from core.onnx_infer.vc_inference import OnnxRVC

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


    def split_audio_vad(self, audio_path, target_dir='processed', audio_name=None):
        audio = AudioSegment.from_file(audio_path)
        max_len = len(audio)
        target_folder = os.path.join(target_dir, audio_name)

        start_time = time.perf_counter()
        vad = self.pipeline(audio_path)
        end_time = time.perf_counter()
        print(f'VAD time: {end_time - start_time}')
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

            # filter out the segment shorter than 1.0s and longer than 10s
            save = audio_seg.duration_seconds > 1.0 and \
                    audio_seg.duration_seconds < 10.
            print(audio_seg.duration_seconds)
            if save:
                output_file = os.path.join(wavs_folder, fname)
                audio_seg.export(output_file, format='wav')

            if k < len(segments) - 1:
                start_time = max(0, segments[k+1].start - 0.08)

            s_ind = s_ind + 1
        return wavs_folder
    
    def split_audio_all(self, audio_path, target_dir='processed', audio_name=None):
        audio = AudioSegment.from_file(audio_path)
        max_len = len(audio)
        target_folder = os.path.join(target_dir, audio_name)

        # create directory
        os.makedirs(target_folder, exist_ok=True)
        wavs_folder = os.path.join(target_folder, 'wavs')
        os.makedirs(wavs_folder, exist_ok=True)

        audio_seg = audio[0:max_len]
        # segment file name
        fname = f"{audio_name}_seg0.wav"
        output_file = os.path.join(wavs_folder, fname)
        audio_seg.export(output_file, format='wav')
        return wavs_folder
    
    def split_audio_timestamp(self, audio_path, timestamps, target_dir='processed', audio_name=None):
        audio = AudioSegment.from_file(audio_path)
        max_len = len(audio)
        target_folder = os.path.join(target_dir, audio_name)

        # create directory
        os.makedirs(target_folder, exist_ok=True)
        wavs_folder = os.path.join(target_folder, 'wavs')
        os.makedirs(wavs_folder, exist_ok=True)

        for s_ind, (start, end) in enumerate(timestamps):
            if s_ind == 0:
                start_time = max(0, start)
            end_time = end
            audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]
            # segment file name
            fname = f"{audio_name}_seg{s_ind}.wav"
            # filter out the segment shorter than 1.0s and longer than 10s
            save = audio_seg.duration_seconds > 1.0 and \
                    audio_seg.duration_seconds < 10.
            print(audio_seg.duration_seconds)
            if save:
                output_file = os.path.join(wavs_folder, fname)
                audio_seg.export(output_file, format='wav')
            if s_ind < len(timestamps) - 1:
                start_time = max(0, timestamps[s_ind+1][0] - 0.08)

        return wavs_folder

    def get_se(self, audio_path, vc_model, target_dir='processed', vad=True, timestamps=None):
        device = vc_model.device
        version = vc_model.version
        print("OpenVoice version:", version)

        audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{se_extractor.hash_numpy_array(audio_path)}"
        se_path = os.path.join(target_dir, audio_name, 'se.pth')

        if not timestamps:
            if vad:
                wavs_folder = self.split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
        elif timestamps == 'all':
            wavs_folder = self.split_audio_all(audio_path, target_dir=target_dir, audio_name=audio_name)
        else:
            wavs_folder = self.split_audio_timestamp(audio_path, timestamps, target_dir=target_dir, audio_name=audio_name)

        from glob import glob
        audio_segs = glob(f'{wavs_folder}/*.wav')
        if len(audio_segs) == 0:
            # raise NotImplementedError('No audio segments found!')
            print('No audio segments found!, use origin tts audio')
            return None
        
        return vc_model.extract_se(audio_segs, se_save_path=se_path)

    def convert(self, source_wav, target_wav, source_timestamps=None, target_timestamps=None):
        start = time.perf_counter()
        # TODO: source wav 都是TTS語者，可以共用se節省時間
        # TODO: target wav 截到差不多5秒就可以停止了，比較real time，應該是要先用perf_counter看看 get_se 的 bottleneck是哪幾行程式
        source_se = self.get_se(source_wav, self.tone_color_converter, vad=True, timestamps=source_timestamps)
        target_se = self.get_se(target_wav, self.tone_color_converter, vad=True, timestamps=target_timestamps)
        end = time.perf_counter()
        print(f'Get source and target se: {end - start:.4f}s')

        save_path = source_wav.replace('_azure_temp', '_final_conversion')
        # save_path = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/final_conversion.wav'
        if source_se is None or target_se is None:
            return source_wav

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
    
class RVCConverter(ConverterBase):
    def __init__(self):
        super().__init__()
        self.customs_model_with_name = {
            'evonne': "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/core/onnx_infer/onnx_weights/evo-20240524_e35_s1575.onnx", 
            'laura': "/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/core/onnx_infer/onnx_weights/laura-test.onnx"
        }
        
        self.sampling_rate = 40000  # 采样率
        self.get_all_custom_models(self.customs_model_with_name)

    def get_all_custom_models(self, customs_model_with_name):
        self.name2model = {}
        vec_name = (
            "vec-768-layer-12"  # 内部自动补齐为 f"pretrained/{vec_name}.onnx" 需要onnx的vec模型
        )
        hop_size = 512
        for name, path in customs_model_with_name.items():
            print(f"Loading model {name} from {path}")
            model = OnnxRVC(
                path, vec_path=vec_name, sr=self.sampling_rate, hop_size=hop_size, device="cuda"
            )
            self.name2model[name] = model

    def convert(self, wav_path, model_name):
        if model_name not in self.name2model:
            print(f"Model {model_name} not found!")
            return None
        save_path = wav_path.replace('_azure_temp', '_final_conversion')
        f0_up_key = 0  # 升降调
        sid = 0  # 角色ID
        f0_method = "dio"  # F0提取算法
        model = self.name2model[model_name]
        audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)
        soundfile.write(save_path, audio, self.sampling_rate)
        return save_path