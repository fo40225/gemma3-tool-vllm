vllm v0.18.1

vllm serve unsloth/medgemma-1.5-4b-it \
 --chat-template tool_chat_template_medgemma15.jinja \
 --enable-auto-tool-choice \
 --tool-parser-plugin medgemma15_tool_parser.py \
 --tool-call-parser medgemma15 \
 --reasoning-parser-plugin medgemma15_reasoning_parser.py \
 --reasoning-parser medgemma15

parser also work on gemma3 or medgemma

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

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "messages": [
    {"role": "user", "content": "Case Scenario: A 72-year-old male with a history of hypertension and heart failure presents to the ER with confusion and profound weakness. He has been taking a water pill for his heart condition but recently increased the dose due to swelling in his ankles. Laboratory Results:  * Serum Sodium (Na+): 128 mEq/L (Normal: 135-145) * Serum Glucose: 540 mg/dL (Normal: 70-100) * Serum Potassium: 3.2 mEq/L (Normal: 3.5-5.0) * Blood Pressure: 110/70 mmHg  Question: What is the most appropriate next step in managing this patients sodium level, and why is a simple low sodium diagnosis misleading here?"}
  ]
}'

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "messages": [
    {"role": "user", "content": "Case Scenario: A 72-year-old male with a history of hypertension and heart failure presents to the ER with confusion and profound weakness. He has been taking a water pill for his heart condition but recently increased the dose due to swelling in his ankles. Laboratory Results:  * Serum Sodium (Na+): 128 mEq/L (Normal: 135-145) * Serum Glucose: 540 mg/dL (Normal: 70-100) * Serum Potassium: 3.2 mEq/L (Normal: 3.5-5.0) * Blood Pressure: 110/70 mmHg  Question: What is the most appropriate next step in managing this patients sodium level, and why is a simple low sodium diagnosis misleading here?"}
  ],
  "chat_template_kwargs": {"enable_thinking": false}
}'
