"""
Model-specific interfaces for handling different LLM chat templates and formatting.

This package provides abstractions for working with different language models,
each of which may have different chat templates, message formatting, and
special tokens.

Usage:
    from models import create_model_interface

    # Create interface for a specific model
    interface = create_model_interface("Qwen/Qwen2.5-7B-Instruct")

    # Use interface to get model-specific information
    system_msg = interface.get_system_message()
    messages = interface.build_messages(question, answer)
    assistant_prefix = interface.get_assistant_prefix()
"""

from .base import ModelInterface, AsyncModelBackend, ForwardResult, GenerationResult
from .qwen3 import Qwen3ModelInterface
from .granite import GraniteModelInterface
from .factory import create_model_interface, create_model_backend
from .hf_backend import HuggingFaceBackend
from .vllm_backend import VLLMBackend

__all__ = [
    'ModelInterface',
    'Qwen3ModelInterface',
    'GraniteModelInterface',
    'create_model_interface',
    'AsyncModelBackend',
    'ForwardResult',
    'GenerationResult',
    'HuggingFaceBackend',
    'VLLMBackend',
    'create_model_backend',
]
