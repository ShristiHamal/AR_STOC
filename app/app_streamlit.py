

# app/app_streamlit.py
import streamlit as st
from PIL import Image
import tempfile
from app.controlnet_inference import run_controlnet_inference

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="AR Smart Try-On", layout="wide")
st.title("AR Smart Try-On")

mode = st.radio("Input mode:", ["Upload Image"])
prompt = st.text_input("Prompt", value="Person wearing clothing with photorealistic style")
steps = st.slider("Inference steps", 1, 50, 8)

st.sidebar.header("Select Clothing")
uploaded_cloth = st.sidebar.file_uploader("Upload clothing image", type=["png","jpg","jpeg"])
uploaded_person = st.file_uploader("Upload person image", type=["png","jpg","jpeg"])

if st.button("Run Try-On"):
    if not uploaded_person or not uploaded_cloth:
        st.error("Upload both person and clothing images.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            person_path = f"{tmpdir}/person.png"
            cloth_path = f"{tmpdir}/cloth.png"
            Image.open(uploaded_person).convert("RGB").save(person_path)
            Image.open(uploaded_cloth).convert("RGB").save(cloth_path)

            with st.spinner("Running inference..."):
                try:
                    out_im = run_controlnet_inference(person_path, cloth_path, prompt=prompt, num_inference_steps=steps)
                    st.image(out_im, caption="Try-On Result", use_column_width=True)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
