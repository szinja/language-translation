# Streamlit Inference App to Demo

import streamlit as st
from transformers import MarianTokenizer, MarianMTModel

st.set_page_config(page_title="English → isiXhosa Translator")

@st.cache_resource
def load_model(model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    return tokenizer, model

st.title("English → isiXhosa Translator")
st.markdown("Translate English sentences into isiXhosa using your fine-tuned Transformer model.")

model_path = "./en-xh-model"
tokenizer, model = load_model(model_path)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text_input = st.text_area("Enter English text", height=100)

if st.button("Translate"):
    if text_input.strip() != "":
        xh_translation = translate(text_input)
        st.success(f"**isiXhosa Translation:**\n\n{xh_translation}")
    else:
        st.warning("Please enter some text.")
