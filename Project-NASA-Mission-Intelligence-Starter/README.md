# NASA Mission Intelligence — Offline RAG Stack
(No clouds. No keys. No telemetry. Just metal.)

## 1. Model Drop
DeepSeek‑R1‑Distill‑Qwen‑14B‑Uncensored.Q4_K_M.gguf  
https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-Uncensored-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf

Put it somewhere sane:

~/LocalModels/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf

## 2. KoboldCpp Boot Sequence
Local engine, Vulkan push, OpenAI‑shim enabled:

koboldcpp-linux-x64 \
  --model ~/LocalModels/DeepSeek-R1-Distill-Qwen-14B-Uncensored.Q4_K_M.gguf \
  --usevulkan \
  --gpulayers 999 \
  --contextsize 4096

Ports:
- OpenAI‑style: http://localhost:5001/v1
- Kobold API:   http://localhost:5001/api/
- Web UI:       http://localhost:5001/lcpp/

Leave it running. It’s the reactor core.

## 3. LLM Client Wiring
llm_client.py points everything at the local endpoint:

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:5001/v1"
)

The pipeline builds:
- system prompt
- retrieved context
- conversation history
- user query
Then fires it into chat.completions.create().

## 4. Retrieval Engine (ChromaDB)
rag_client.py handles:
- backend discovery
- collection loading
- vector search
- context formatting

All embeddings and documents live locally in chroma_db_openai/.

## 5. Chat Console (Streamlit)
Launch the operator panel:

streamlit run chat.py

The UI:
- loads the Chroma collection
- retrieves top‑k NASA mission chunks
- injects them into the prompt
- queries the local LLM
- displays the answer

Zero external calls. Everything stays on your machine.

## 6. Batch Evaluation (Offline)
Run the evaluator:

python run_batch_eval.py

Uses:
- your retriever
- your local LLM
- simple_evaluator.py (custom, no RAGAS dependency)

Outputs:
- response_relevancy
- faithfulness
- aggregate metrics

Typical run with the 14B model:
response_relevancy ≈ 0.70  
faithfulness ≈ 0.62

Good enough to show the RAG loop is alive and doing its job.

## 7. Layout
chat.py                # Streamlit chat UI  
llm_client.py          # Local OpenAI‑shim client  
rag_client.py          # ChromaDB loader + retriever  
embedding_pipeline.py  # Embedding + ingestion  
simple_evaluator.py    # Offline evaluator  
run_batch_eval.py      # Batch scoring  
chroma_db_openai/      # Local vector store  
evaluation_dataset.txt # Questions for scoring

## 8. Operating Notes
- Entire stack is offline.  
- No API keys.  
- No cloud inference.  
- No telemetry.  
- Model can leak <think> traces; expected for uncensored builds.  
- Retrieval + trimming keeps context under 4096 tokens.  

## 9. Local Hardware Profile
This stack runs on a 2016‑era desktop that refuses to die:

- Intel Core i7‑6700K (Skylake, 4C/8T @ 4.3 GHz)
- 16 GB DDR4
- Intel Arc B580 (Battlemage) — Vulkan compute, primary GPU
- NVIDIA GTX 970 — secondary card, display duties
- Dual NVMe SSDs + 1 TB HDD
- Linux Mint 22.3 (Ubuntu 24.04 base), kernel 6.17
- Xorg session for stable multi‑GPU behavior

KoboldCpp pushes the 14B Q4_K_M model through Vulkan on the Arc card.  
The GTX 970 handles the monitors so the compute path stays clean.  
ChromaDB sits on NVMe and responds instantly.  

## 10. Assistance and Tools
Development used VS Codium, the Continue extension, and a local DeepSeek model for code assistance along with some structure help from MS:Copilot.
Final design and implementation were written and validated manually.

## This is a self‑contained RAG machine built for local execution, no dependencies on external LLMs or hosted services.
