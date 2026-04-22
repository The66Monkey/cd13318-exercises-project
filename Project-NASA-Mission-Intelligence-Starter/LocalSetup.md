# Local LLM Setup (KoboldCpp)

Offline stack. No API keys. No cloud calls.

## Model

DeepSeek‑R1‑Distill‑Qwen‑14B‑Uncensored.Q4_K_M.gguf  
Direct link:  
https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-Uncensored-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf

Drop it somewhere predictable:

```
~/LocalModels/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf
```

## KoboldCpp Boot

Fire up the model with Vulkan + full offload + OpenAI‑API shim:

```bash
koboldcpp-linux-x64 \
  --model ~/LocalModels/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf \
  --usevulkan \
  --gpulayers 999 \
  --contextsize 4096 \
  --api
```

Endpoints exposed:

- OpenAI‑compatible → `http://localhost:5001/v1`
- Kobold API → `http://localhost:5001/api/`
- Web UI → `http://localhost:5001/lcpp/`

Leave this terminal running.

## LLM Client Wiring

`llm_client.py` points everything at the local endpoint:

```python
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:5001/v1"
)
```

The RAG pipeline builds:

- system prompt  
- retrieved context  
- conversation history  
- user query  

Then sends it straight into `chat.completions.create()`.

## Chat UI

Launch the Streamlit front-end:

```bash
streamlit run chat.py
```

The UI:

- loads your ChromaDB collection  
- retrieves NASA mission chunks  
- injects context into the prompt  
- queries the local LLM  
- shows the answer + optional RAGAS metrics  

Everything stays offline.  
Nothing leaves your machine.