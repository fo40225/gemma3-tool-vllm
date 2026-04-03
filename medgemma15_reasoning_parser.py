from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Role label that MedGemma15 emits at the start of the thinking channel.
# The model generates: <|channel>thought\n...reasoning...<channel|>
# This prefix must be stripped to expose only the actual reasoning content.
_THOUGHT_PREFIX = "thought\n"


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought\n`` role label from the beginning of text.
    """
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX) :]
    return text


class MedGemma15ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MedGemma 1.5 thinking models.

    MedGemma 1.5 uses ``<unused94>`` and ``<unused95>`` tokens to delimit
    reasoning content within its output. Thinking mode is activated by
    passing ``enable_thinking=True`` in the chat template kwargs.

    Output pattern when thinking is enabled::

        <unused94>thought
        ...chain of thought reasoning...<unused95>
        Final answer text here.

    The ``thought\n`` role label inside the thinking delimiters is a
    structural artefact (analogous to ``user\n`` in ``<start_of_turn>user\n...``).
    This parser strips it so that downstream consumers see only the
    actual reasoning text.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<unused94>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "<unused95>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Instance state for streaming prefix stripping.
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False

        # Check if thinking is enabled via chat template kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning content, stripping the ``thought\n`` role label."""
        # Strip <unused94> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            if not self.thinking_enabled:
                # Thinking explicitly disabled — treat everything as content.
                return None, model_output
            # Thinking enabled but no <unused95>: output was truncated.
            # Everything generated so far is reasoning.
            return model_output, None

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)

        # Strip the thought\n label from reasoning
        if reasoning:
            reasoning = _strip_thought_label(reasoning)

        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract streaming reasoning, stripping ``thought\n`` from the
        first reasoning delta(s).
        """
        # When thinking is disabled, no think tokens appear in the output.
        if not self.thinking_enabled:
            return DeltaMessage(content=delta_text)

        # Strip <unused94> from delta if present (in old template or edge case
        # where the model generates <unused94> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                # Strip thought\n prefix from reasoning
                if reasoning:
                    reasoning = _strip_thought_label(reasoning)
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        elif self.end_token_id in previous_token_ids:
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        else:
            # No end token yet: still in reasoning phase.
            # Strip thought\n prefix from first reasoning delta
            reasoning_text = delta_text
            if not self._prefix_stripped:
                reasoning_text = _strip_thought_label(reasoning_text)
                if reasoning_text != delta_text:
                    self._prefix_stripped = True
            return DeltaMessage(reasoning=reasoning_text)

from vllm.reasoning import ReasoningParserManager

ReasoningParserManager.register_module(
    name="medgemma15", module=MedGemma15ReasoningParser, force=True
)
