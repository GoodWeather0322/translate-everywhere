import edge_tts
from edge_tts import VoicesManager
import random

class TTSBase:
    def __init__(self):
        pass

    def speak(self, text):
        raise NotImplementedError
    
class EdgeTTS(TTSBase):
    def __init__(self):
        super().__init__()

    async def dynamic_voice_selection(self, gender="Male", language="zh"):
        voice_manager = await VoicesManager.create()
        voices = voice_manager.find(Gender=gender, Language=language)
        voice = random.choice(voices)["Name"]
        voice = 'zh-TW-HsiaoChenNeural'
        return voice
    
    async def lang_voice_sellection(self, lang):
        voice_dict = {
            'zh' : 'zh-TW-HsiaoChenNeural',
            'en' : 'en-US-AvaNeural', 
            'ja' : 'ja-JP-KeitaNeural', 
            'ko' : 'ko-KR-HyunsuNeural'
        }
        if lang not in voice_dict:
            print('lang not support at this time')
            return False
        
        return voice_dict[lang]

    async def speak(self, text, voice, output):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output)

    async def tts_flow(self, language, text):
        voice = await self.lang_voice_sellection(language)
        temp_file = '/mnt/disk1/chris/uaicraft_workspace/translate-everywhere/test_code/temp.wav'
        await self.speak(text, voice, temp_file)

        return temp_file