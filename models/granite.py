"""
Granite model-specific interface implementation.

Granite models (granite-3.1-8b-instruct, etc.) from IBM also use ChatML format
similar to Qwen models.
"""

from typing import List, Dict
from .base import ModelInterface


class GraniteModelInterface(ModelInterface):
    """
    Model interface for IBM Granite models.

    Granite 3.1 models use ChatML format similar to Qwen:
    - System message: <|im_start|>system\n{content}<|im_end|>
    - User message: <|im_start|>user\n{content}<|im_end|>
    - Assistant message: <|im_start|>assistant\n{content}<|im_end|>
    """

    def get_system_message(self) -> str:
        """
        Get the default system message for Granite models.

        Returns:
            System message string
        """
        return "You are a helpful assistant."

    def get_assistant_prefix(self) -> str:
        """
        Get the ChatML assistant prefix used by Granite models.

        Returns:
            Assistant prefix string in ChatML format
        """
        return "<|im_start|>assistant\n"

    def build_messages(self, question: str, answer: str) -> List[Dict[str, str]]:
        """
        Build the standard message structure for Granite models.

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
