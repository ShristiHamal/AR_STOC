# app_streamlit.py
import streamlit as st
from PIL import Image
from pathlib import Path
import tempfile
from app.controlnet_inference import run_controlnet_inference

st.set_page_config(page_title="AR Smart Try-On", layout="wide")

st.title("AR Smart Try-On (Streamlit)")

uploaded_person = st.file_uploader("Upload person image", type=["jpg","jpeg","png"])
uploaded_mask = st.file_uploader("Upload cloth mask image (512x512 white = mask)", type=["png","jpg","jpeg"])

prompt = st.text_input("Prompt", value="Person wearing the clothing with photorealistic style")
steps = st.slider("Inference steps", 1, 50, 8)

if st.button("Run Try-On"):
    if not uploaded_person or not uploaded_mask:
        st.error("Please upload both person and mask images.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            person_path = Path(tmpdir) / "person.jpg"
            mask_path = Path(tmpdir) / "mask.png"
            Image.open(uploaded_person).convert("RGB").save(person_path)
            Image.open(uploaded_mask).convert("RGB").save(mask_path)

            with st.spinner("Running inference (may take a while)..."):
                try:
                    out_im = run_controlnet_inference(str(person_path), str(mask_path), prompt=prompt, num_inference_steps=steps)
                    st.image(out_im, caption="Try-On Result", use_column_width=True)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
