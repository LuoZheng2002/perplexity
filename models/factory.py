"""
Factory function to create the appropriate model interface based on model name.
"""

from .base import ModelInterface
from .qwen import QwenModelInterface
from .granite import GraniteModelInterface


def create_model_interface(model_name: str) -> ModelInterface:
    """
    Create the appropriate model interface based on the model name.

    This function examines the model name and returns the corresponding
    ModelInterface implementation. It supports:
    - Qwen models (Qwen/Qwen2.5-*)
    - Granite models (ibm-granite/granite-*)

    Args:
        model_name: Hugging Face model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        ModelInterface instance appropriate for the given model

    Raises:
        ValueError: If the model name is not recognized

    Examples:
        >>> interface = create_model_interface("Qwen/Qwen2.5-7B-Instruct")
        >>> isinstance(interface, QwenModelInterface)
        True

        >>> interface = create_model_interface("ibm-granite/granite-3.1-8b-instruct")
        >>> isinstance(interface, GraniteModelInterface)
        True
    """
    model_name_lower = model_name.lower()

    # Check for Qwen models
    if "qwen" in model_name_lower:
        return QwenModelInterface()

    # Check for Granite models
    if "granite" in model_name_lower:
        return GraniteModelInterface()

    # If no match found, raise an error with helpful message
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Supported models are: Qwen (Qwen/*), Granite (ibm-granite/granite-*). "
        f"To add support for this model, create a new ModelInterface implementation "
        f"in the models/ directory and update the factory function."
    )
