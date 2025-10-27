import streamlit as st
from PIL import Image
import cv2
import tempfile
from app.controlnet_inference import run_controlnet_inference

st.set_page_config(page_title="AR Smart Try-On", layout="wide")
st.title("AR Smart Try-On (Webcam Live Preview)")

mode = st.radio("Choose input:", ["Upload Image", "Webcam Live"])
prompt = st.text_input("Prompt", value="Person wearing the clothing with photorealistic style")
steps = st.slider("Inference steps", 1, 50, 8)

if mode == "Upload Image":
    uploaded_person = st.file_uploader("Upload person image", type=["jpg","jpeg","png"])
    uploaded_mask = st.file_uploader("Upload cloth mask image (512x512)", type=["png","jpg","jpeg"])
    
    if st.button("Run Try-On"):
        if not uploaded_person or not uploaded_mask:
            st.error("Please upload both images.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                person_path = f"{tmpdir}/person.jpg"
                mask_path = f"{tmpdir}/mask.png"
                Image.open(uploaded_person).convert("RGB").save(person_path)
                Image.open(uploaded_mask).convert("RGB").save(mask_path)
                with st.spinner("Running inference..."):
                    try:
                        out_im = run_controlnet_inference(person_path, mask_path, prompt=prompt, num_inference_steps=steps)
                        st.image(out_im, caption="Try-On Result", use_column_width=True)
                    except Exception as e:
                        st.error(f"Inference failed: {e}")

elif mode == "Webcam Live":
    st.warning("Webcam mode is experimental.")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    uploaded_mask = st.file_uploader("Upload cloth mask image", type=["png","jpg","jpeg"])
    
    run_live = st.checkbox("Run live try-on")
    while run_live and uploaded_mask:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        mask_img = Image.open(uploaded_mask).convert("RGB").resize((512,512))
        try:
            out_frame = run_controlnet_inference(frame_pil, mask_img, prompt=prompt, num_inference_steps=steps)
            FRAME_WINDOW.image(out_frame)
        except Exception as e:
            st.error(f"Live inference failed: {e}")
            break
    cap.release()
