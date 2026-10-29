# StackVerse: AI-Powered Tech Stack Recommender

**Live Demo:** [https://stackverse.streamlit.app/](https://stackverse.streamlit.app/)  
**Fine-Tuned Model:** [`Enaksheexe/stackverse-llm`](https://huggingface.co/Enaksheexe/stackverse-llm)

StackVerse is a specialized recommendation tool designed to help developers select the right frontend, backend, database, and deployment tools based on specific project requirements, constraints, and scalability targets. 

Instead of relying on generic prompt wrappers around proprietary APIs, I built this application on top of a custom fine-tuned `Qwen 2.5 3B` model (`Enaksheexe/stackverse-llm`) trained specifically for software architecture trade-offs and stack selection.

---

## Overview & Motivation

Choosing a tech stack early in a project often defaults to whatever is currently trending rather than what fits the architecture. General-purpose LLMs tend to give overly verbose, non-committal answers when asked for stack recommendations. 

I built this project to solve that by fine-tuning a lightweight 3B parameter model to output structured, practical architecture recommendations—balancing developer velocity, ecosystem maturity, and infrastructure overhead.

---

## Model Fine-Tuning Pipeline

I fine-tuned the base `Qwen/Qwen2.5-3B` model to create [`Enaksheexe/stackverse-llm`](https://huggingface.co/Enaksheexe/stackverse-llm) using the following workflow:

* **Training Environment:** Google Colab (T4 GPU instance)
* **Optimization Framework:** [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and ~60% reduced VRAM usage.
* **Fine-Tuning Method:** QLoRA (4-bit quantization via `bitsandbytes`) with PEFT adapters targeting all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Training Loop:** Hugging Face `TRL` (`SFTTrainer`) formatted with structured instruction-response pairs covering diverse software requirements (real-time systems, MVP SaaS, high-throughput APIs, edge deployments, etc.).
* **Model Hosting:** Merged adapters and pushed weights directly to the Hugging Face Hub under `Enaksheexe/stackverse-llm`.

---

## Tech Stack

* **Model:** `Enaksheexe/stackverse-llm` (Fine-tuned Qwen 2.5 3B)
* **ML / Training:** Unsloth, PyTorch, Hugging Face `transformers`, `TRL`, `PEFT`, Google Colab
* **Frontend & Serving:** Streamlit (Python)
* **Deployment:** Streamlit Community Cloud

---

## Running Locally

### 1. Clone the repository
```bash
git clone [https://github.com/enaksheexe/tech-stack-recommender-app.git](https://github.com/enaksheexe/tech-stack-recommender-app.git)
cd tech-stack-recommender-app
