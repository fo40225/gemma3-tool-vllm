vllm v0.18.1

vllm serve unsloth/medgemma-1.5-4b-it \
 --chat-template tool_chat_template_medgemma15.jinja \
 --enable-auto-tool-choice \
 --tool-parser-plugin medgemma15_tool_parser.py \
 --tool-call-parser medgemma15 \
 --reasoning-parser-plugin medgemma15_reasoning_parser.py \
 --reasoning-parser medgemma15

tool parser also work on gemma3 or medgemma

dgx station v100
sudo docker run \
    -d --restart unless-stopped \
    -v /home/user/.cache:/root/.cache \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    --entrypoint bash vllm/vllm-openai:v0.18.1 \
 -c '
curl -L -O https://raw.githubusercontent.com/fo40225/gemma3-tool-vllm/refs/heads/main/tool_chat_template_medgemma15.jinja
curl -L -O https://raw.githubusercontent.com/fo40225/gemma3-tool-vllm/refs/heads/main/medgemma15_tool_parser.py
vllm serve \
    --model unsloth/medgemma-27b-it \
    --served-model-name unsloth/medgemma-27b-it \
    --max-model-len 131072 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --mm-processor-cache-gb 0 \
 --chat-template tool_chat_template_medgemma15.jinja \
 --enable-auto-tool-choice \
 --tool-parser-plugin medgemma15_tool_parser.py \
 --tool-call-parser medgemma15 \
'
