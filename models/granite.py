"""
Granite model-specific interface implementation.

Granite models (granite-3.1-8b-instruct, etc.) from IBM also use ChatML format
similar to Qwen models.
"""

from typing import List, Dict, Any
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

    def build_messages_for_perplexity_forward(self, tokenizer: Any, question: str, answer: str,
                                             language: str) -> str:
        """
        Build the message structure for perplexity calculation (forward pass) and apply chat template.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer: The assistant's answer
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string after applying chat template
        """
        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def build_messages_for_perplexity_generate(self, tokenizer: Any, question: str,
                                              language: str) -> str:
        """
        Build the message structure for answer generation and apply chat template.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string after applying chat template with generation prompt
        """
        # Build language-specific instructions
        if language.lower() == "english":
            instruction = "Please answer the question in English with a concise phrase instead of a complete sentence. Start with an uncapitalized first word."
        else:
            instruction = f"Please answer the question in {language} with a concise phrase instead of a complete sentence."

        # Combine question with instruction
        user_content = f"{question}\n\n{instruction}"

        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": user_content}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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
        # do not use prefill
        return formatted

    def build_messages_for_compare_cot(self, tokenizer: Any, question: str,
                                           answer1: str, answer2: str) -> str:
        """
        Build the message structure for comparison with reasoning and apply chat template.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string after applying chat template
        """
        prompt = f"""Given the following question and two answers, which answer is better?

Question: {question}

Answer 1: {answer1}
Answer 2: {answer2}

Please briefly explain your reasoning, and then provide your final decision in the following format:
\\boxed{{X}} where X is either 1 or 2."""

        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": prompt}
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
