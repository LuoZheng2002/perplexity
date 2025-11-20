from enum import Enum, auto
from dataclasses import dataclass

from typing import NamedTuple, Union

class Model(Enum):
    GRANITE_3_1_8B_INSTRUCT = "ibm-granite/granite-3.1-8b-instruct"
    QWEN_2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_2_5_14B_INSTRUCT = "Qwen/Qwen2.5-14B-Instruct"
    QWEN_2_5_32B_INSTRUCT = "Qwen/Qwen2.5-32B-Instruct"
    QWEN_2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"

class ResultType(Enum):
    PREFERENCE_DIRECT = auto()
    PREFERENCE_THINKING = auto()
    PERPLEXITY = auto()

@dataclass(frozen=True)
class Config:
    model: Model
    lang1: str
    lang2: str
    subject: str
    result_type: ResultType


configs = [
    Config(Model.QWEN_2_5_7B_INSTRUCT, "zh_cn", "en", "philosophy", ResultType.PREFERENCE_DIRECT),
    Config(Model.QWEN_2_5_7B_INSTRUCT, "zh_cn", "en", "philosophy", ResultType.PERPLEXITY),
    Config(Model.QWEN_2_5_7B_INSTRUCT, "zh_cn", "en", "philosophy", ResultType.PREFERENCE_THINKING),    
]