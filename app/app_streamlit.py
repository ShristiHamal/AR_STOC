import streamlit as st
from PIL import Image

# Correct import path
from app.controlnet_inference import run_controlnet_tryon


#config
st.set_page_config(
    page_title="AR Virtual Try-On",
    layout="wide"
)

st.title("AR Virtual Try-On System)")


#person image upload
st.subheader("Upload Person Image")

person_file = st.file_uploader(
    "Select a person image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if person_file:
    person_img = Image.open(person_file).convert("RGB")
    st.image(person_img, caption="Person Image", width=300)
else:
    person_img = None


#cloth image upload
st.subheader("Upload Clothing Image")

cloth_file = st.file_uploader(
    "Select a clothing image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if cloth_file:
    cloth_img = Image.open(cloth_file).convert("RGB")
    st.image(cloth_img, caption="Cloth Image", width=300)
else:
    cloth_img = None


#inference
st.subheader("Inference Settings")

col1, col2 = st.columns(2)

with col1:
    num_steps = st.slider("Inference Steps", 10, 75, 30)

with col2:
    guidance = st.slider("Guidance Scale", 1.0, 12.0, 7.5)

prompt = st.text_input(
    "Prompt",
    "a realistic virtual try-on of the person wearing the garment"
)



run_btn = st.button(" Run Virtual Try-On")


# Execute inference on button click
if run_btn:
    if not (person_img and cloth_img):
        st.error("Please upload BOTH a person image and a cloth image.")
    else:
        with st.spinner("Generating try-on result... please wait ⏳"):
            try:
                output = run_controlnet_tryon(
                person_image=person_img,
                cloth_image=cloth_img,
                mask_image=None,  # auto cloth mask
                pose_json=None,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                prompt=prompt,
            )

                st.success("Done!")
                st.image(output, caption="Virtual Try-On Result", use_column_width=True)

            except Exception as e:
                st.error(f" Inference failed: {e}")
