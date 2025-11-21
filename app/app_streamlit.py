import streamlit as st
from PIL import Image
from inpaint_inference import run_inpaint_tryon


# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="AR Virtual Try-On  Interactive Wardrobe",
    layout="wide",
)

st.title(" AR Virtual Try-On  Interactive Wardrobe")


# ------------------------------------------------------------
# Sidebar – Person Image (Upload or Webcam)
# ------------------------------------------------------------
st.sidebar.header("1. Person Photo")

person_source = st.sidebar.radio(
    "Choose source",
    ["Upload image", "Use webcam"],
)

person_image = None

if person_source == "Upload image":
    person_file = st.sidebar.file_uploader(
        "Upload a person image",
        type=["jpg", "jpeg", "png"],
        key="person_uploader",
    )
    if person_file:
        person_image = Image.open(person_file).convert("RGB")

else:
    cam_img = st.sidebar.camera_input("Take a photo", key="camera_input")
    if cam_img:
        person_image = Image.open(cam_img).convert("RGB")

if person_image:
    st.sidebar.image(person_image, caption="Person Image", use_column_width=True)
else:
    st.sidebar.info("Please upload or take a person photo.")


# ------------------------------------------------------------
# Layout Columns
# ------------------------------------------------------------
col_clothes, col_controls, col_output = st.columns([1.2, 0.8, 1.4])


# ------------------------------------------------------------
# Column 1 – Clothing Upload Panel
# ------------------------------------------------------------
with col_clothes:
    st.subheader("2. Upload Clothing Images")
    clothes_files = st.file_uploader(
        "Upload multiple clothing images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="clothes_uploader",
    )

    if not clothes_files:
        st.info("Upload clothing items to build your wardrobe.")
        selected_cloth_image = None
        selected_cloth_name = None
    else:
        st.success(f"{len(clothes_files)} clothing items uploaded.")

        # Dropdown for selecting a clothing item
        cloth_names = [f.name for f in clothes_files]
        selected_cloth_name = st.selectbox(
            "Select a clothing item to try on",
            cloth_names,
            key="cloth_selector",
        )

        # Find selected clothing file
        selected_cloth_image = None
        for f in clothes_files:
            if f.name == selected_cloth_name:
                selected_cloth_image = Image.open(f).convert("RGB")
                break

        if selected_cloth_image:
            st.image(selected_cloth_image, caption="Selected Clothing")


# ------------------------------------------------------------
# Clothing Gallery – MUST be outside columns (Streamlit rule)
# ------------------------------------------------------------
if clothes_files:
    st.write("### Clothing Gallery")
    gallery_cols = st.columns(3)

    for idx, cloth_file in enumerate(clothes_files):
        with gallery_cols[idx % 3]:
            cloth_img = Image.open(cloth_file).convert("RGB")
            st.image(cloth_img, caption=cloth_file.name)


# ------------------------------------------------------------
# Column 2 – Try-On Controls
# ------------------------------------------------------------
with col_controls:
    st.subheader("3. Try-On Settings")

    num_inference_steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=75,
        value=25,
        step=5,
    )

    prompt = st.text_area(
        "Optional prompt",
        value="A realistic virtual try-on.",
    )

    run_btn = st.button(" Run Virtual Try-On")


# ------------------------------------------------------------
# Column 3 – Output Section
# ------------------------------------------------------------
with col_output:
    st.subheader("4. Virtual Try-On Result")

    if run_btn:
        # Validate inputs
        if person_image is None:
            st.error("Please upload a person image first.")
        elif not clothes_files:
            st.error("Please upload at least one clothing item.")
        elif selected_cloth_image is None:
            st.error("Please select a clothing item from the list.")
        else:
            with st.spinner("Generating virtual try-on result…"):
                try:
                    result = run_inpaint_tryon(
                    person_image,
                    selected_cloth_image,
                    num_inference_steps=num_inference_steps,
                    prompt=prompt,
)

                    st.image(result, caption=f"Result: {selected_cloth_name}")

                except Exception as e:
                    st.error(f" Inference failed: {e}")

    else:
        st.info("Click **Run Virtual Try-On** to generate the result.")
