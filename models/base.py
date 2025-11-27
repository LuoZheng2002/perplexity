"""
Base abstract class for model-specific interfaces.

This module defines the abstract interface that all model-specific implementations
must follow to handle chat templates, message formatting, and token position finding.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import asyncio


class ModelInterface(ABC):
    """
    Abstract base class for model-specific behavior.

    Different language models use different chat templates and formatting conventions.
    This interface abstracts those differences to allow the perplexity calculation
    and preference collection logic to work across different models.
    """

    @abstractmethod
    def get_system_message(self) -> str:
        """
        Get the system message content for the model.

        Returns:
            System message string
        """
        pass

    @abstractmethod
    def get_assistant_prefix(self) -> str:
        """
        Get the assistant prefix token/string used in the chat template.

        This is the string that appears before the assistant's response in the
        formatted conversation. For example, in ChatML format used by Qwen models,
        this would be "<|im_start|>assistant\n".

        Returns:
            Assistant prefix string
        """
        pass

    @abstractmethod
    def build_messages_for_perplexity_forward(self, tokenizer: Any, question: str, answer: str,
                                             language: str) -> str:
        """
        Build the message structure for perplexity calculation (forward pass) and apply chat template.

        The prompt includes language-specific instructions to encourage the model to
        respond in the target language with concise phrasing. For English, an
        uncapitalized first word is encouraged. This variant includes the answer in
        the messages and uses add_generation_prompt=False for perplexity calculation.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer: The assistant's answer
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string after applying chat template
        """
        pass

    @abstractmethod
    def build_messages_for_perplexity_generate(self, tokenizer: Any, question: str,
                                              language: str) -> str:
        """
        Build the message structure for answer generation and apply chat template.

        The prompt includes language-specific instructions to encourage the model to
        respond in the target language with concise phrasing. For English, an
        uncapitalized first word is encouraged. This variant does NOT include the answer
        and uses add_generation_prompt=True to prompt the model to generate its own answer.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            language: Formal language name (e.g., "Chinese", "English")

        Returns:
            Formatted conversation string after applying chat template with generation prompt
        """
        pass

    @abstractmethod
    def build_messages_for_compare_directly(self, tokenizer: Any, question: str,
                                           answer1: str, answer2: str) -> str:
        """
        Build the message structure for direct comparison and apply chat template.

        This method builds a prompt asking the model to compare two answers,
        applies the chat template with add_generation_prompt=True.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string after applying chat template without prefill
        """
        pass

    @abstractmethod
    def build_messages_for_compare_cot(self, tokenizer: Any, question: str,
                                           answer1: str, answer2: str) -> str:
        """
        Build the message structure for comparison with reasoning and apply chat template.

        This method builds a prompt asking the model to compare two answers with
        reasoning/explanation before giving the final answer. Unlike the direct version,
        this encourages the model to think through its decision. Applies the chat template
        with add_generation_prompt=True.

        Args:
            tokenizer: The model's tokenizer
            question: The user's question
            answer1: The first answer to compare
            answer2: The second answer to compare

        Returns:
            Formatted conversation string after applying chat template
        """
        pass

    def find_answer_start(self, tokenizer: Any, full_ids: List[int],
                         answer_tokens: List[int]) -> int:
        """
        Find the starting position of the answer tokens in the full sequence.

        This default implementation uses the assistant prefix to locate where
        the answer begins. Models can override this if they need custom logic.

        Args:
            tokenizer: The model's tokenizer
            full_ids: List of all token IDs in the full conversation
            answer_tokens: List of token IDs for just the answer

        Returns:
            Index of the first answer token in full_ids

        Raises:
            ValueError: If the answer start position cannot be found
        """
        assistant_prefix = self.get_assistant_prefix()
        prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids

        # Find the prefix in the full sequence
        L = len(prefix_ids)
        for i in range(len(full_ids) - L + 1):
            if full_ids[i:i+L] == prefix_ids:
                return i + L  # Return position after the prefix

        raise ValueError(
            f"Assistant prefix '{assistant_prefix}' not found in tokenized conversation. "
            f"This might indicate a mismatch between the model interface and the actual model."
        )


class ForwardResult:
    """Result from a forward pass inference."""
    def __init__(self, logits: Any, input_ids: List[int]):
        """
        Args:
            logits: Tensor of shape [seq_len, vocab_size] containing logits for each position
            input_ids: List of token IDs that were input to the model
        """
        self.logits = logits
        self.input_ids = input_ids


class GenerationResult:
    """Result from a text generation inference."""
    def __init__(
        self,
        generated_text: str,
        generated_ids: List[int],
        full_text: str,
        full_ids: List[int],
        logits: Optional[Any] = None
    ):
        """
        Args:
            generated_text: The decoded generated text (without input prompt)
            generated_ids: List of token IDs for the generated text only
            full_text: The complete text (prompt + generated)
            full_ids: List of token IDs for the complete sequence (prompt + generated)
            logits: Logits/logprobs for generated tokens (format varies by backend):
                    - HuggingFace: tuple of tensors, one per generated token [vocab_size]
                    - vLLM: list of dicts with logprob information per token
        """
        self.generated_text = generated_text
        self.generated_ids = generated_ids
        self.full_text = full_text
        self.full_ids = full_ids
        self.logits = logits


class AsyncModelBackend(ABC):
    """
    Abstract base class for async model backends that handle inference.

    Different backends (HuggingFace, vLLM) have different batching strategies.
    This interface provides low-level primitives (forward pass, generation) that
    allow concurrent request submission while optimizing batch processing.

    Backends should automatically batch concurrent requests for efficiency.
    """

    @abstractmethod
    async def forward_async(
        self,
        formatted_prompt: str,
        max_length: int = 2048
    ) -> ForwardResult:
        """
        Asynchronously run forward pass on a formatted prompt.

        This method accepts a single request and returns logits for the sequence.
        The backend may batch multiple concurrent requests internally for efficiency.

        Args:
            formatted_prompt: The formatted prompt text (after applying chat template)
            max_length: Maximum sequence length for tokenization

        Returns:
            ForwardResult containing logits and input_ids
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        formatted_prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        do_sample: bool = False
    ) -> GenerationResult:
        """
        Asynchronously generate text from a formatted prompt.

        This method accepts a single request and returns generated text.
        The backend may batch multiple concurrent requests internally for efficiency.

        Args:
            formatted_prompt: The formatted prompt text (after applying chat template)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for greedy decoding)
            do_sample: Whether to use sampling

        Returns:
            GenerationResult containing generated text and token IDs
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Cleanup resources and shutdown the backend.

        Should be called when inference is complete to properly release resources.
        """
        pass
