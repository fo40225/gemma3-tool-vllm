from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

class MedGemma15ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MedGemma 1.5 model.

    The MedGemma 1.5 model uses <unused94>...<unused95> tokens to denote
    reasoning content text. This parser extracts the reasoning content from
    the model output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<unused94>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<unused95>"

from vllm.reasoning import ReasoningParserManager

ReasoningParserManager.register_module(
    name="medgemma15", module=MedGemma15ReasoningParser, force=True
)
