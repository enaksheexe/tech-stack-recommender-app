import os
import json
import urllib.request
import urllib.error
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="StackVerse | AI Tech Stack Recommender",
    page_icon="🛠️",
    layout="centered"
)

# Header
st.title("StackVerse: AI Tech Stack Architect")
st.caption("Powered by fine-tuned model: `Enaksheexe/stackverse-llm` | TiesVerse")

# Automatically load token from Streamlit Secrets
default_token = ""
groq_key = ""
try:
    if "HF_TOKEN" in st.secrets:
        default_token = str(st.secrets["HF_TOKEN"]).strip()
    if "GROQ_API_KEY" in st.secrets:
        groq_key = str(st.secrets["GROQ_API_KEY"]).strip()
except Exception:
    default_token = os.getenv("HF_TOKEN", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

# Sidebar for Model Info
with st.sidebar:
    st.header("Project Details")
    if default_token or groq_key:
        hf_token = default_token
        st.success("API Connected & Ready")
    else:
        hf_token = st.text_input(
            "API Token (HF or Groq)",
            type="password",
            help="Paste your hf_... or gsk_... token here or in Streamlit Secrets"
        ).strip()
        st.markdown("---")

    st.markdown(
        "**Fine-Tuned Model:**\n"
        "[Enaksheexe/stackverse-llm](https://huggingface.co/Enaksheexe/stackverse-llm)\n\n"
        "**Dataset:**\n"
        "[Enaksheexe/enakshee_tech_stack_dataset](https://huggingface.co/datasets/Enaksheexe/enakshee_tech_stack_dataset)"
    )

# Input Form
st.subheader("Tell us about your project")

with st.form("stack_form"):
    col1, col2 = st.columns(2)

    with col1:
        project_category = st.selectbox(
            "Project Category",
            ["Web Application", "Mobile App", "AI / Machine Learning", "E-Commerce", "SaaS Platform", "Real-Time App", "Other"]
        )
        team_size = st.selectbox(
            "How many people? (Team Size)",
            ["Solo Developer (1)", "Small Team (2-4)", "Medium Team (5-10)", "Large Team (10+)"]
        )

    with col2:
        budget = st.selectbox(
            "What is your budget?",
            ["Free Tier Only ($0)", "Low / Bootstrap (<$50/mo)", "Moderate ($50-$500/mo)", "Enterprise / Flexible"]
        )
        timeline = st.selectbox(
            "How much time? (Timeline)",
            ["Hackathon (24-48 hours)", "MVP (1-4 weeks)", "Standard (1-3 months)", "Long-term (6+ months)"]
        )

    specific_use_case = st.text_area(
        "What is your project? (Specific Use Case)",
        placeholder="e.g., Real-time collaborative code editor for college students with live chat...",
        height=100
    )

    submit_button = st.form_submit_button("Generate Stack", use_container_width=True)

# Format Prompt in Alpaca Style
alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert Software Architect AI trained on the StackVerse dataset (Enaksheexe/stackverse-llm). Analyze the user's project details, team size, budget, and timeline, and recommend the optimal Frontend, Backend, Database, and Deployment tech stack with clear reasoning.

### Input:
Project Category: {category} | Specific Use Case: {use_case} | Team Size: {team} | Budget: {budget} | Timeline: {timeline}

### Response:
"""

def call_openai_compatible(url, token, model_name, messages):
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "model": model_name,
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.3
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

def get_available_groq_models(token):
    preferred = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            live_ids = [
                m["id"] for m in data.get("data", [])
                if not any(skip in m["id"].lower() for skip in ["whisper", "guard", "orpheus", "tts", "embed"])
            ]
            ordered = [m for m in preferred if m in live_ids]
            others = [m for m in live_ids if m not in ordered]
            return ordered + others
    except Exception:
        return preferred

if submit_button:
    if not specific_use_case.strip():
        st.warning("Please describe what your project is before generating a stack!")
    elif not (hf_token or groq_key):
        st.warning("Please add your token in Streamlit Secrets.")
    else:
        with st.spinner("Consulting StackVerse AI Architect..."):
            prompt = alpaca_template.format(
                category=project_category,
                use_case=specific_use_case,
                team=team_size,
                budget=budget,
                timeline=timeline
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are StackVerse (Enaksheexe/stackverse-llm), an expert Software Architect AI. Always format your response clearly with Recommended Stack (Frontend, Backend, Database, Deployment/Hosting) and concise architectural reasoning."
                },
                {"role": "user", "content": prompt}
            ]

            result = None
            errors = []

            # Route A: Dynamically fetch and call unlocked Groq models
            active_gsk = groq_key if groq_key else (hf_token if hf_token.startswith("gsk_") else "")
            if active_gsk:
                groq_models = get_available_groq_models(active_gsk)
                for g_model in groq_models:
                    try:
                        result = call_openai_compatible(
                            "https://api.groq.com/openai/v1/chat/completions",
                            active_gsk,
                            g_model,
                            messages
                        )
                        if result:
                            break
                    except urllib.error.HTTPError as e:
                        err_body = e.read().decode("utf-8", errors="ignore")
                        errors.append(f"Groq ({g_model}): {e.code} - {err_body}")
                    except Exception as e:
                        errors.append(f"Groq ({g_model}): {e}")

            # Route B: Fallback to HF Router
            if not result and hf_token and not hf_token.startswith("gsk_"):
                hf_router_models = [
                    "openai/gpt-oss-120b:fastest",
                    "openai/gpt-oss-20b:fastest",
                    "Qwen/Qwen2.5-7B-Instruct:fastest",
                    "Qwen/Qwen2.5-72B-Instruct:fastest"
                ]
                for h_model in hf_router_models:
                    try:
                        result = call_openai_compatible(
                            "https://router.huggingface.co/v1/chat/completions",
                            hf_token,
                            h_model,
                            messages
                        )
                        if result:
                            break
                    except urllib.error.HTTPError as e:
                        err_body = e.read().decode("utf-8", errors="ignore")
                        errors.append(f"HF Router ({h_model}): {e.code} - {err_body}")
                    except Exception as e:
                        errors.append(f"HF Router ({h_model}): {e}")

            if result:
                st.success("Optimal Tech Stack Generated!")
                st.markdown("### 🛠️ Recommended Architecture")
                st.markdown(result)
            else:
                st.error("Could not reach inference server. Details:\n\n" + "\n\n".join(errors[:2]))
