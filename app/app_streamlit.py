import streamlit as st
from PIL import Image

# Correct import path
from inpaint_inference import run_inpaint_tryon


st.set_page_config(page_title="AR Try-On", layout="wide")
st.title(" AR Virtual Try-On ")


# -------------------------------
# PERSON IMAGE
# -------------------------------
person_file = st.file_uploader(
    "Upload Person Image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
)

if person_file:
    person_img = Image.open(person_file).convert("RGB")
    st.image(person_img, caption="Person Image", width=300)
else:
    person_img = None


# -------------------------------
# CLOTH IMAGE
# -------------------------------
cloth_file = st.file_uploader(
    "Upload Clothing Image (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
)

if cloth_file:
    cloth_img = Image.open(cloth_file).convert("RGB")
    st.image(cloth_img, caption="Cloth Image", width=300)
else:
    cloth_img = None


# -------------------------------
# MASK IMAGE (REQUIRED)
# -------------------------------
mask_file = st.file_uploader(
    "Upload Mask Image (PNG, white=inpaint)",
    type=["png"],
)

if mask_file:
    mask_img = Image.open(mask_file).convert("L")
    st.image(mask_img, caption="Mask Image", width=300)
else:
    mask_img = None


# -------------------------------
# Inference Settings
# -------------------------------
num_steps = st.slider("Inference Steps", 10, 75, 25)
guidance = st.slider("Guidance Scale", 1.0, 12.0, 7.5)
prompt = st.text_input("Prompt", "a realistic virtual try-on of the person wearing the garment")

run_btn = st.button("Run Try-On")


# -------------------------------
# RUN INFERENCE
# -------------------------------
if run_btn:
    if not (person_img and cloth_img and mask_img):
        st.error("Please upload person image, cloth image, AND mask image.")
    else:
        with st.spinner("Running inpainting model..."):
            try:
                output = run_inpaint_tryon(
                    person_image=person_img,
                    cloth_image=cloth_img,
                    mask_image=mask_img,
                    pose_json=None,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    prompt=prompt,
                )

                st.success("Done!")
                st.image(output, caption="Try-On Result", use_column_width=True)

            except Exception as e:
                st.error(f" Inference failed: {e}")
