import os
import json
import urllib.request
import streamlit as st
from huggingface_hub import InferenceClient

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
        default_token = st.secrets["HF_TOKEN"]
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    default_token = os.getenv("HF_TOKEN", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

# Sidebar for Model Info
with st.sidebar:
    st.header("Project Details")
    if default_token or groq_key:
        hf_token = default_token
        st.success("API Connected & Ready")
    else:
        hf_token = st.text_input(
            "API Token",
            type="password",
            help="Paste your token in Streamlit Secrets"
        )
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
            last_error = None

            # 1. If a direct Groq key or gsk_ token is provided, use fast direct endpoint
            active_gsk = groq_key if groq_key else (hf_token if hf_token.startswith("gsk_") else "")
            if active_gsk:
                try:
                    req = urllib.request.Request(
                        "https://api.groq.com/openai/v1/chat/completions",
                        data=json.dumps({
                            "model": "llama-3.3-70b-versatile",
                            "messages": messages,
                            "max_tokens": 600,
                            "temperature": 0.3
                        }).encode("utf-8"),
                        headers={
                            "Authorization": f"Bearer {active_gsk}",
                            "Content-Type": "application/json",
                            "User-Agent": "StackVerse/1.0"
                        }
                    )
                    with urllib.request.urlopen(req) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        result = data["choices"][0]["message"]["content"]
                except Exception as e:
                    last_error = e

            # 2. Otherwise try Hugging Face Router (works once Groq key is linked in HF Settings)
            if not result and hf_token and not hf_token.startswith("gsk_"):
                client = InferenceClient(token=hf_token)
                models_to_try = [
                    "meta-llama/Llama-3.3-70B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct",
                    "meta-llama/Llama-3.1-8B-Instruct"
                ]
                for model_id in models_to_try:
                    try:
                        response = client.chat_completion(
                            model=model_id,
                            messages=messages,
                            max_tokens=600,
                            temperature=0.3
                        )
                        result = response.choices[0].message.content
                        break
                    except Exception as e:
                        last_error = e
                        continue

            if result:
                st.success("Optimal Tech Stack Generated!")
                st.markdown("### 🛠️ Recommended Architecture")
                st.markdown(result)
            else:
                st.error(f"Error connecting to inference server: {last_error}")
