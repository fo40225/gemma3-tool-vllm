import ast
import json

from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser

# State machine states
STATE_NORMAL = "NORMAL"
STATE_TOOL = "TOOL"

class MedGemma15ToolParser(ToolParser):
    """
    Tool call parser for MedGemma 1.5 model.

    The MedGemma 1.5 model produces tool calls wrapped with ```tool_code and ```.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.backtick_token = self.vocab.get("```", None)
        self.tool_token = self.vocab.get("tool", None)
        self.underscore_token = self.vocab.get("_", None)
        self.code_token = self.vocab.get("code", None)

        self._state = STATE_NORMAL
        self._pending: list[int] = []
        self._tool_buffer: list[int] = []

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if "```tool_code" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[ToolCall] = []

        # Extract content between ```tool_code and ``` markers
        blocks = []
        block_ranges = []  # Track start/end positions of each block
        start_token = "```tool_code"
        end_token = "```"
        current_start = None
        i = 0

        while i < len(model_output):
            if model_output.startswith(start_token, i):
                current_start = i + len(start_token)
                i += len(start_token)
            elif model_output.startswith(end_token, i):
                if current_start is not None:
                    blocks.append(model_output[current_start:i])
                    block_ranges.append((current_start - len(start_token), i + len(end_token)))
                    current_start = None
                i += len(end_token)
            else:
                i += 1

        # Try to parse each block and track which ones succeed
        successful_block_indices = set()
        for idx, block_content in enumerate(blocks):
            try:
                parsed = self._parse_tool_content(block_content)
                if parsed:
                    tool_calls.extend(parsed)
                    successful_block_indices.add(idx)
            except Exception:
                continue

        # Build content by excluding successfully parsed tool_code blocks
        content = []
        last_end = 0

        for idx, (block_start, block_end) in enumerate(block_ranges):
            if idx not in successful_block_indices:
                # Keep failed blocks in content
                content.append(model_output[last_end:block_start])
                content.append(model_output[block_start:block_end])
                last_end = block_end
            else:
                # Skip successful blocks
                content.append(model_output[last_end:block_start])
                last_end = block_end

        # Add remaining text after last block
        content.append(model_output[last_end:])

        final_content = "".join(content)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=final_content
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from streaming output using simple state machine.
        Trigger: ```tool_code (4 tokens: backtick, tool, underscore, code)
        End marker: ``` (single backtick token)
        """
        for token_id in delta_token_ids:
            if self._state == STATE_TOOL:
                # In TOOL mode, buffer tokens until we see ```
                if token_id == self.backtick_token:
                    # End of tool block
                    self._state = STATE_NORMAL
                    result = self._process_tool_buffer()
                    # If tool_calls is empty, keep the current token for next iteration
                    if result and not result.tool_calls:
                        self._pending = [token_id]
                    else:
                        self._pending = []
                    self._tool_buffer = []
                    return result
                else:
                    self._tool_buffer.append(token_id)
                    return None
            else:  # STATE_NORMAL
                self._pending.append(token_id)
                # Try to match the trigger: ```tool_code
                if self._is_trigger_match(self._pending):
                    # Enter tool mode
                    self._state = STATE_TOOL
                    # Remove the 4 trigger tokens from pending
                    self._pending = []
                    # Clear tool buffer
                    self._tool_buffer = []
                    return None
                elif self._is_trigger_prefix(self._pending):
                    # Still waiting for more tokens, keep pending
                    return None
                else:
                    # Not a valid prefix, flush the pending as content
                    content = self.model_tokenizer.decode(self._pending)
                    self._pending = []
                    return DeltaMessage(content=content)
        return None

    def _is_trigger_match(self, tokens):
        """Check if tokens exactly match the trigger ```tool_code."""
        if len(tokens) != 4:
            return False
        return (tokens[0] == self.backtick_token and
                tokens[1] == self.tool_token and
                tokens[2] == self.underscore_token and
                tokens[3] == self.code_token)

    def _is_trigger_prefix(self, tokens):
        """Check if tokens match a prefix of the trigger ```tool_code."""
        if len(tokens) > 4:
            return False
        if len(tokens) >= 1 and tokens[0] != self.backtick_token:
            return False
        if len(tokens) >= 2 and tokens[1] != self.tool_token:
            return False
        if len(tokens) >= 3 and tokens[2] != self.underscore_token:
            return False
        if len(tokens) >= 4 and tokens[3] != self.code_token:
            return False
        return True

    def _process_tool_buffer(self) -> DeltaMessage | None:
        if not self._tool_buffer:
            return DeltaMessage(content=f"```tool_code")
        content = self.model_tokenizer.decode(self._tool_buffer)
        parsed = self._parse_tool_content(content)
        if parsed:
            base_index = len(self.prev_tool_call_arr)

            for t in parsed:
                self.prev_tool_call_arr.append({
                    "name": t.function.name,
                    "arguments": t.function.arguments,
                })

            if len(self.streamed_args_for_tool) < len(self.prev_tool_call_arr):
                self.streamed_args_for_tool.append("")

            self.current_tool_id = len(self.prev_tool_call_arr) - 1

            return DeltaMessage(tool_calls=[
                self._tool_call_to_delta_tool(t, base_index + i)
                for i, t in enumerate(parsed)
            ])

        return DeltaMessage(content=f"```tool_code{content}")

    def _tool_call_to_delta_tool(self, tool_call, index):
        return DeltaToolCall(
            id=tool_call.id,
            type=tool_call.type,
            index=index,
            function=DeltaFunctionCall(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments
            ).model_dump(exclude_none=True)
        )

    def _parse_tool_content(self, content):
        """Parse tool content from JSON or AST."""

        # First try full JSON parse
        try:
            obj = json.loads(content)
            tool_calls = self._parse_json_tools(obj)
            if tool_calls:
                return tool_calls
        except:
            pass

        # Then try line-by-line JSON parse
        results = []
        for line in content.strip().splitlines():
            try:
                obj = json.loads(line)
            except:
                continue
            tool_calls = self._parse_json_tools(obj)
            if tool_calls:
                results.extend(tool_calls)
        if results:
            return results

        # Fallback to AST parsing
        try:
            node = ast.parse(content, mode="exec")
            results = []
            if node.body:
                for stmt in node.body:
                    tool_calls = self._extract_tools_from_node(stmt, unwrap_print=True)
                    if tool_calls:
                        results.extend(tool_calls)
            if results:
                return results
        except:
            pass

        return []

    def _parse_json_tools(self, obj):
        """Parse JSON tools (single object, JSONL, or array)."""
        result = []

        to_parse = []
        if isinstance(obj, dict):
            to_parse = [obj]
        elif isinstance(obj, list):
            to_parse = obj

        for item in to_parse:
            if isinstance(item, dict):
                name = item.get("name")
                params = item.get("parameters", item.get("arguments", {}))
                if name:
                    if not isinstance(params, dict):
                        params = {}

                    tool_call_kwargs = {}
                    tool_call_kwargs["id"] = make_tool_call_id(
                            id_type="random",
                            func_name=name,
                            idx=None,
                        )

                    result.append(ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(params, ensure_ascii=False)
                        ),
                        **tool_call_kwargs
                    ))

        return result

    def _extract_tools_from_node(self, node, unwrap_print=True):
        """
        Extract tools from a single AST node (not a list).
        This should be called with individual nodes like Expr, Call, List, etc.
        """

        # Check if node is a dict with only (name, parameters) or (name, arguments)
        if isinstance(node, ast.Dict):
            keys = [self._ast_value_to_dict(k) for k in node.keys]
            values = [self._ast_value_to_dict(v) for v in node.values]
            if len(keys) == 2 and set(keys) in ({'name', 'parameters'}, {'name', 'arguments'}):
                name = keys[keys.index('name')]
                params_key = 'parameters' if 'parameters' in keys else 'arguments'
                params_idx = keys.index(params_key)
                params = values[params_idx]
                if not isinstance(params, dict):
                    params = {}
                tool_call_kwargs = {}
                tool_call_kwargs["id"] = make_tool_call_id(
                    id_type="random",
                    func_name=name,
                    idx=None,
                )
                return [ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(params, ensure_ascii=False)
                    ),
                    **tool_call_kwargs
                )]

        # 1. Container nodes: iterate elements (handles List, Tuple, Set, etc.)
        if hasattr(node, 'elts'):
            result = []
            for elt in node.elts:
                result.extend(self._extract_tools_from_node(elt, unwrap_print))
            return result

        # 2. Call node: extract tool or unwrap special cases
        if isinstance(node, ast.Call):
            # Get function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                return []  # non-simple function name

            # Check if there's only one argument and it's a Dict
            if len(node.args) == 1 and isinstance(node.args[0], ast.Dict):
                # Parse dict as tool - use func_name as function name, dict keys as arguments
                arg = node.args[0]
                keys = [self._ast_value_to_dict(k) for k in arg.keys]
                values = [self._ast_value_to_dict(v) for v in arg.values]
                args = {}
                for k, v in zip(keys, values):
                    args[k] = v
                if args:
                    # Generate tool call ID using the actual function name
                    tool_call_kwargs = {}
                    tool_call_kwargs["id"] = make_tool_call_id(
                        id_type="random",
                        func_name=func_name,
                        idx=None,
                    )
                    return [ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(args, ensure_ascii=False)
                        ),
                        **tool_call_kwargs
                    )]

            # Print special case: unwrap if any arg is a Call
            if unwrap_print and func_name == "print" and node.args:
                results = []
                for arg in node.args:
                    if isinstance(arg, ast.Call):
                        results.extend(self._extract_tools_from_node(arg, unwrap_print=False))
                if results:
                    return results

            # Default: generate tool call from kwargs
            args = {}
            for kw in node.keywords:
                v = self._ast_value_to_dict(kw.value)
                args[kw.arg] = v

            # Generate tool call ID
            tool_call_kwargs = {}
            tool_call_kwargs["id"] = make_tool_call_id(
                id_type="random",
                func_name=func_name,
                idx=None,
            )

            return [ToolCall(
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(args, ensure_ascii=False)
                ),
                **tool_call_kwargs
            )]

        # 3. Generic recursion: nodes with 'value' attribute
        if hasattr(node, 'value'):
            return self._extract_tools_from_node(node.value, unwrap_print)

        return []

    def _ast_value_to_dict(self, node):
        """Convert AST node to Python value."""
        # 1. Terminal node: Constant
        if isinstance(node, ast.Constant):
            return node.value

        # 2. Container nodes: Dict, List, Tuple, Set (any node with 'elts' or 'keys/values')
        if hasattr(node, 'elts'):
            return [self._ast_value_to_dict(v) for v in node.elts]
        if hasattr(node, 'keys') and hasattr(node, 'values'):
            return {self._ast_value_to_dict(k): self._ast_value_to_dict(v)
                    for k, v in zip(node.keys, node.values)}

        # 3. Generic recursion: nodes with 'value' attribute
        if hasattr(node, 'value'):
            return self._ast_value_to_dict(node.value)

        # 4. Unary operators
        if hasattr(node, 'operand') and hasattr(node, 'op'):
            val = self._ast_value_to_dict(node.operand)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
            return val

        return None


from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

ToolParserManager.register_module(
        name="medgemma15", module=MedGemma15ToolParser, force=True
)
