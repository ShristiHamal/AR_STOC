import streamlit as st
from PIL import Image
import cv2
import tempfile
from controlnet_inference import run_controlnet_inference

st.set_page_config(page_title="AR Smart Try-On", layout="wide")
st.title("AR Smart Try-On")

# --- UI: mode selection ---
mode = st.radio("Choose input:", ["Upload Image", "Webcam Live"])
prompt = st.text_input("Prompt", value="Person wearing the clothing with photorealistic style")
steps = st.slider("Inference steps", 1, 50, 8)

# --- Clothing selection ---
st.sidebar.header("Select Clothing")
uploaded_cloth = st.sidebar.file_uploader("Upload clothing image (mask)", type=["png","jpg","jpeg"])
# Optionally, you can have predefined clothes:
# cloth_options = ["dress1.png", "jacket.png", "shirt.png"]
# selected_cloth = st.sidebar.selectbox("Choose cloth", cloth_options)

if mode == "Upload Image":
    uploaded_person = st.file_uploader("Upload person image", type=["jpg","jpeg","png"])
    
    if st.button("Run Try-On"):
        if not uploaded_person or not uploaded_cloth:
            st.error("Please upload both person and clothing images.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                person_path = f"{tmpdir}/person.jpg"
                mask_path = f"{tmpdir}/mask.png"
                Image.open(uploaded_person).convert("RGB").save(person_path)
                Image.open(uploaded_cloth).convert("RGB").save(mask_path)
                
                with st.spinner("Running inference..."):
                    try:
                        out_im = run_controlnet_inference(person_path, mask_path, prompt=prompt, num_inference_steps=steps)
                        st.image(out_im, caption="Try-On Result", use_column_width=True)
                    except Exception as e:
                        st.error(f"Inference failed: {e}")

elif mode == "Webcam Live":
    st.warning("Webcam live mode is experimental and may be slow.")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    run_live = st.checkbox("Run live try-on")
    
    if uploaded_cloth is None:
        st.info("Upload a clothing mask to run live try-on.")
    
    while run_live and uploaded_cloth:
        ret, frame = cap.read()
        if not ret: break
        
        # Convert webcam frame to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Prepare mask/clothing
        mask_img = Image.open(uploaded_cloth).convert("RGB").resize((512,512))
        
        # Run ControlNet inference
        try:
            out_frame = run_controlnet_inference(frame_pil, mask_img, prompt=prompt, num_inference_steps=steps)
            FRAME_WINDOW.image(out_frame)
        except Exception as e:
            st.error(f"Live inference failed: {e}")
            break
    
    cap.release()
