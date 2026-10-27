import streamlit as st
from huggingface_hub import InferenceClient

# Page Configuration
st.set_page_config(
    page_title="StackVerse | AI Tech Stack Recommender",
    page_icon="💻",
    layout="centered"
)

# Header
st.title("StackVerse: AI Tech Stack Architect")
st.caption("Powered by fine-tuned model: `Enaksheexe/stackverse-llm` | TiesVerse")

# Sidebar for Hugging Face Token & Model Info
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input(
        "Hugging Face Read Token",
        type="password",
        help="Paste a READ token from huggingface.co/settings/tokens"
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

    submit_button = st.form_submit_button(" Generate Stack", use_container_width=True)

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
    elif not hf_token:
        st.warning("Please enter a Hugging Face Read Token in the left sidebar (or Streamlit Secrets) to run inference.")
    else:
        with st.spinner("Consulting StackVerse AI Architect..."):
            try:
                # Initialize Hugging Face Inference Client
                client = InferenceClient(token=hf_token)

                prompt = alpaca_template.format(
                    category=project_category,
                    use_case=specific_use_case,
                    team=team_size,
                    budget=budget,
                    timeline=timeline
                )

                # Calls the Qwen2.5-3B-Instruct serverless endpoint with Alpaca prompt structure
                response = client.chat_completion(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are StackVerse (Enaksheexe/stackverse-llm), an expert Software Architect AI. Always format your response clearly with Recommended Stack (Frontend, Backend, Database, Deployment/Hosting) and concise architectural reasoning."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.3
                )

                result = response.choices[0].message.content
                st.success("Optimal Tech Stack Generated!")
                st.markdown("### 🛠️ Recommended Architecture")
                st.markdown(result)

            except Exception as e:
                st.error(f"Error connecting to Hugging Face: {e}")
