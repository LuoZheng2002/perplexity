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

from .base import ModelInterface
from .qwen import QwenModelInterface
from .granite import GraniteModelInterface
from .factory import create_model_interface

__all__ = [
    'ModelInterface',
    'QwenModelInterface',
    'GraniteModelInterface',
    'create_model_interface',
]
