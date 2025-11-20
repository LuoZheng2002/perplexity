"""
Qwen model-specific interface implementation.

Qwen models (Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, etc.) use the ChatML
format for chat conversations.
"""

from typing import List, Dict
from .base import ModelInterface


class QwenModelInterface(ModelInterface):
    """
    Model interface for Qwen models.

    Qwen models use ChatML format with the following structure:
    - System message: <|im_start|>system\n{content}<|im_end|>
    - User message: <|im_start|>user\n{content}<|im_end|>
    - Assistant message: <|im_start|>assistant\n{content}<|im_end|>
    """

    def get_system_message(self) -> str:
        """
        Get the default system message for Qwen models.

        Returns:
            System message string
        """
        return "You are a helpful assistant."

    def get_assistant_prefix(self) -> str:
        """
        Get the ChatML assistant prefix used by Qwen models.

        Returns:
            Assistant prefix string in ChatML format
        """
        return "<|im_start|>assistant\n"

    def build_messages(self, question: str, answer: str) -> List[Dict[str, str]]:
        """
        Build the standard message structure for Qwen models.

        Args:
            question: The user's question
            answer: The assistant's answer

        Returns:
            List of message dictionaries with system, user, and assistant roles
        """
        return [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
