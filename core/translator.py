from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

class TranslatorBase:
    def __init__(self):
        pass

    def translate(self, text):
        raise NotImplementedError

class LLMTranslator(TranslatorBase):
    def __init__(self):
        super().__init__()

        self.chat_model = ChatOllama(
            base_url='http://localhost:11434',
            model='llama3',
            temperature=0,
        )

        self.multilingual_prompt_dict = {
            "en" : 'How is the weather today', 
            'ja' : '今日の天気はどうですか', 
            'zh' : '今天的天氣如何', 
            'ko' : '방법 날씨가 오늘', 
        }

        prompt_texts = [
            """You are a helpful translator and only output the result in json format.\nEvery word should be carefully translated.\nTranslate this from <{source_language}> to <{target_language}>\n""",
            """<{source_language}>:{source_sentence_example}\n""", 
            """<{target_language}>:{target_sentence_example}\n""", 
            """<{source_language}>:{source_sentence}\n""", 
        ]

        prompt_templates = []
        for i, text in enumerate(prompt_texts):
            if i == 0:
                prompt_templates.append(SystemMessagePromptTemplate.from_template(text))
            else:
                case_number = int((i + 1) / 2)
                if i % 2 == 1:
                    prompt_templates.append(
                        HumanMessagePromptTemplate.from_template(
                            f"{text}"
                        )
                    )
                else:
                    prompt_templates.append(
                        AIMessagePromptTemplate.from_template(
                            f"{text}"
                        )
                    )

        self.chat_template = ChatPromptTemplate.from_messages(prompt_templates)

    def translate(self, prompt_messages):
        generation = self.chat_model.generate(prompt_messages)
        single_generation = generation.generations[0]
        llm_output = single_generation[0].text
        translate_text = llm_output.split(':')[1]
        return translate_text

    def translate_flow(self, source_language, target_language, text):
        if source_language not in self.multilingual_prompt_dict:
            print('source_language not support at this time')
            return False
        if target_language not in self.multilingual_prompt_dict:
            print('target_language not support at this time')
            return False

        prompt_messages = []

        source_sentence_example = self.multilingual_prompt_dict['zh']
        target_sentence_example = self.multilingual_prompt_dict['ja']
        prompt_message = self.chat_template.format_prompt(
            source_language=source_language, 
            target_language=target_language, 
            source_sentence_example=source_sentence_example, 
            target_sentence_example=target_sentence_example, 
            source_sentence=text
        )

        prompt_messages.append(prompt_message.to_messages())

        translate_text = self.translate(prompt_messages)
        print(f'[{source_language}] -> [{target_language}] : {translate_text}')
        
        return translate_text


class AzureTranslator(TranslatorBase):

    def __init__(self):
        super().__init__()