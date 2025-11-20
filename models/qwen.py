"""
Qwen model-specific interface implementation.

Qwen models (Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, etc.) use the ChatML
format for chat conversations.
"""

from typing import List, Dict, Any
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

    def build_messages_for_perplexity(self, tokenizer: Any, question: str, answer: str) -> str:
        """
        Build the message structure for perplexity calculation and apply chat template.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer: The assistant's answer

        Returns:
            Formatted conversation string after applying chat template
        """
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def build_messages_for_compare_directly(self, tokenizer: Any, question: str,
                                           answer1: str, answer2: str) -> str:
        """
        Build the message structure for direct comparison and apply chat template.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string after applying chat template with "\\box{" appended
        """
        prompt = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Provide your judgment IMMEDIATELY without reasoning or explanation. Provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": prompt}
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted + "\\box{"
