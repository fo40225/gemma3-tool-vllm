
vllm v0.18.1

vllm serve unsloth/medgemma-1.5-4b-it \
 --chat-template tool_chat_template_medgemma15.jinja \
 --enable-auto-tool-choice \
 --tool-parser-plugin medgemma15_tool_parser.py \
 --tool-call-parser medgemma15 \
 --reasoning-parser-plugin medgemma15_reasoning_parser.py \
 --reasoning-parser medgemma15

tool parser also work on gemma3 or medgemma
