from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

class ApiModel(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5"
    DEEPSEEK_CHAT = "deepseek-chat"
    LLAMA_3_1_8B = "meta.llama3-1-8b-instruct-v1:0"
    LLAMA_3_1_70B = "meta.llama3-1-70b-instruct-v1:0"

class LocalModel(Enum):
    GRANITE_3_1_8B_INSTRUCT = "ibm-granite/granite-3.1-8b-instruct"
    QWEN_2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_2_5_14B_INSTRUCT = "Qwen/Qwen2.5-14B-Instruct"
    QWEN_2_5_32B_INSTRUCT = "Qwen/Qwen2.5-32B-Instruct"
    QWEN_2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"

# @dataclass
# class LocalModelStruct:
#     model: LocalModel
#     generator: any = None  # Placeholder for the actual generator object

Model = Union[ApiModel, LocalModel]

class Language(Enum):
    CHINESE = auto()
    HINDI = auto()

class TranslateOption(Enum):
    FULLY_TRANSLATED = auto()
    FULLY_TRANSLATED_PROMPT_TRANSLATE = auto()
    PARTIALLY_TRANSLATED = auto()
    FULLY_TRANSLATED_POST_PROCESS_DIFFERENT = auto()
    FULLY_TRANSLATED_POST_PROCESS_SAME = auto(),
    FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME = auto(),

class AddNoiseMode(Enum):
    NO_NOISE = auto()
    SYNONYM = auto()
    PARAPHRASE = auto()

@dataclass(frozen=True)
class Translated:
    language: Language
    option: TranslateOption

@dataclass(frozen=True)
class NotTranslated:
    pass

TranslateMode = Union[Translated, NotTranslated]

class PostProcessOption(Enum):
    DONT_POST_PROCESS = 0
    POST_PROCESS_DIFFERENT = 1
    POST_PROCESS_SAME = 2



# class Config:
#     def __init__(self, model: Model, translate_info: TranslateMode, add_noise_mode: AddNoiseMode):
#         self.model = model
#         self.translate_info = translate_info
#         self.add_noise_mode = add_noise_mode

@dataclass(frozen=True)
class Config:
    model: Model
    translate_mode: TranslateMode
    add_noise_mode: AddNoiseMode

configs: list[Config] = [
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),

    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),
    
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.CLAUDE_SONNET, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),    
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_PARTIALLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModelStruct(LocalModel.GRANITE_3_1_8B_INSTRUCT), translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.PARAPHRASE),

    # Config(model=ApiModel.GPT_4O_MINI, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),


    # Config(model=ApiModel.LLAMA_3_1_8B, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=ApiModel.LLAMA_3_1_70B, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),

    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.SYNONYM),
    
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.SYNONYM),

    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.PARAPHRASE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT), add_noise_mode=AddNoiseMode.SYNONYM),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME), add_noise_mode=AddNoiseMode.SYNONYM),
    


    # Config(model=LocalModel.QWEN_2_5_7B_INSTRUCT, translate_mode=Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED), add_noise_mode=AddNoiseMode.NO_NOISE),
    # Config(model=LocalModel.QWEN_2_5_14B_INSTRUCT, translate_mode=NotTranslated(), add_noise_mode=AddNoiseMode.NO_NOISE),
]
# for model in [LocalModel.QWEN_2_5_7B_INSTRUCT, LocalModel.QWEN_2_5_14B_INSTRUCT]:
# for model in [ApiModel.DEEPSEEK_CHAT]:
for model in [ApiModel.LLAMA_3_1_8B, ApiModel.LLAMA_3_1_70B]:
# for model in [ApiModel.GPT_4O_MINI]:
    for translate_mode in [
        NotTranslated(),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
        Translated(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME),
    ]:
        for add_noise_mode in [
            AddNoiseMode.NO_NOISE,
            AddNoiseMode.SYNONYM,
            AddNoiseMode.PARAPHRASE,
        ]:
            configs.append(
                Config(
                    model=model,
                    translate_mode=translate_mode,
                    add_noise_mode=add_noise_mode,
                )
            )


requires_inference_raw = True
requires_inference_json = True
requires_post_processing = True # rephrase parameter values if the raw output has a similar meaning as the ground truth but is not an exact match
requires_evaluation = True
requires_score = True

evaluation_caching = False