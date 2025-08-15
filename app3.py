import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.34.0"])
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import re

# 1. Enhanced Model Loading
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("model_2")
        model = AutoModelForCausalLM.from_pretrained(
            "model_2",
            torch_dtype=torch.float32
        ).to('cpu')
        return model, tokenizer
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        st.stop()

model, tokenizer = load_model()

# 2. Stronger IELTS-specific generation prompts (using your exact commands)
def generate_response(prompt, mode):
    command = {
        "Thesis": f"Write a precise, well-structured IELTS thesis statement for the question: '{prompt}'. Clearly present your position, ensure specificity, and maintain formal academic tone. Avoid generic phrases, unnecessary details, or mentions of authors, instructions, or essay counts.\nThesis:",
        "Argument": f"Provide one clear, formal, and well-supported academic argument in IELTS style for the question: '{prompt}'. Ensure logical flow with clear reasoning and, where appropriate, relevant examples. Maintain formal academic tone, and avoid generic phrases, authors, instructions, or essay counts.\nArgument:",
        "Conclusion": f"Write a concise and impactful IELTS-style conclusion for the question: '{prompt}'. Summarize the main argument, reinforce your position, and maintain a formal academic tone. Avoid introducing new ideas or mentioning authors, instructions, or essay counts.\nConclusion:"
    }[mode]
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=60,
        temperature=0.7,
        top_k=40,
        do_sample=True,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = generator(command, num_return_sequences=1)
    return clean_output(result[0]['generated_text'], command)

# 3. Enhanced Output Cleaning
def clean_output(text, prompt):
    text = text.replace(prompt, "").strip()
    text = re.sub(r'[<>\[\]\|]', '', text)
    text = re.sub(r'^(Yes|No),?', '', text)
    
    # Ensure proper sentence ending
    if text and text[-1] not in ['.', '!', '?']:
        last_punct = max(text.rfind(p) for p in ['.', '!', '?'])
        text = text[:last_punct+1] if last_punct != -1 else text + '.'
    
    return text.split("\n")[0].strip()

# 4. Modern Streamlit UI
st.set_page_config(
    layout="wide",
    page_title="IELTS Writing Pro | Thesis Generator",
    page_icon="✍️"
)

# Sidebar Dashboard
with st.sidebar:
    st.title("IELTS Writing Pro")
    st.markdown("""
    **Your AI-powered writing assistant for:**
    - Thesis statements
    - Strong arguments
    - Impactful conclusions
    
    **Best for topics like:**
    - Education
    - Technology
    - Environment
    - Society
    - Globalization
    
    *Phrase questions clearly for best results*
    """)
    st.divider()
    st.caption("v2.1 | Academic Writing Assistant")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.header("✍️ IELTS Writing Assistant")
    st.caption("Generate perfect academic statements in seconds")
    
    mode = st.radio(
        "Select output type:",
        ["Thesis", "Argument", "Conclusion"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    user_input = st.text_input(
        "Enter your IELTS question:",
        placeholder="e.g., Should governments regulate social media?"
    )
    
    if st.button("Generate Academic Response", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Crafting your perfect response..."):
                try:
                    response = generate_response(user_input, mode)
                    
                    st.markdown(f"""
                    <div style='
                        padding: 20px;
                        border-radius: 10px;
                        background: #fffff;
                        margin: 15px 0;
                        font-size: 16px;
                        border-left: 4px solid #4e79a7;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    '>
                        <h4 style='color: #fffff; margin-top:0;'>{mode}</h4>
                        <p style='margin-bottom:0;'>{response}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Example Questions")
    st.markdown("""
    <div style='background:#fffff; padding:15px; border-radius:8px;'>
    - Is technology making people less social?<br>
    - Should school uniforms be mandatory?<br>
    - Does social media do more harm than good?<br>
    - Are video games harmful to children?<br>
    - Should fast food be taxed?<br>
    - Is space exploration worth the cost?
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("💡 Tip: Use question format for best results, incase of an abnormal response try to refresh the server or generate answer again.")

# Footer
st.divider()

st.caption("© 2025 IELTS Writing Pro | Academic AI Assistant")
