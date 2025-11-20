import streamlit as st
from PIL import Image
import tempfile
import os

# Import your real try-on model
from app.controlnet_inference import run_controlnet_inference

#Create Streamlit UI
st.set_page_config(page_title="AR Virtual Try-On", layout="wide")
st.title("AR Virtual Try-On System")

st.markdown("""
Upload a **person image**, **clothing image**, **clothing mask**, and **OpenPose JSON**
to generate a high-quality AI-powered virtual try-on result.
""")


prompt = st.text_input(
    "Prompt:",
    value="A person wearing the given clothing, photorealistic."
)

steps = st.slider(
    "Inference steps:",
    min_value=5,
    max_value=75,
    value=20,
)


st.sidebar.header("Upload Inputs")

uploaded_person = st.sidebar.file_uploader(
    "Person Image:",
    type=["jpg", "jpeg", "png"]
)

uploaded_cloth = st.sidebar.file_uploader(
    "Clothing Image:",
    type=["jpg", "jpeg", "png"]
)

uploaded_mask = st.sidebar.file_uploader(
    "Clothing Mask Image:",
    type=["png", "jpg"]
)

uploaded_pose_json = st.sidebar.file_uploader(
    "Pose JSON File:",
    type=["json"]
)

if uploaded_person:
    st.sidebar.image(uploaded_person, caption="Person Preview", use_column_width=True)

if uploaded_cloth:
    st.sidebar.image(uploaded_cloth, caption="Clothing Preview", use_column_width=True)

if uploaded_mask:
    st.sidebar.image(uploaded_mask, caption="Mask Preview", use_column_width=True)



# Try-On Execution
if st.button("Run Virtual Try-On"):

    if not (uploaded_person and uploaded_cloth and uploaded_mask and uploaded_pose_json):
        st.error("Please upload ALL required files: Person, Cloth, Mask, and Pose JSON.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            person_path = os.path.join(tmpdir, "person.png")
            cloth_path = os.path.join(tmpdir, "cloth.png")
            mask_path = os.path.join(tmpdir, "mask.png")
            pose_json_path = os.path.join(tmpdir, "pose.json")

            # Save person + cloth
            Image.open(uploaded_person).convert("RGB").save(person_path)
            Image.open(uploaded_cloth).convert("RGB").save(cloth_path)

            # Save mask as binary mask
            mask_img = Image.open(uploaded_mask).convert("L")
            mask_bin = mask_img.point(lambda x: 255 if x > 128 else 0, mode="1")
            mask_bin = mask_bin.convert("L")
            mask_bin.save(mask_path)

            # Save JSON
            with open(pose_json_path, "wb") as f:
                f.write(uploaded_pose_json.read())

            # Run inference
            with st.spinner("Running Virtual Try-On..."):
                try:
                    output_img = run_controlnet_inference(
                        person_img_path=person_path,
                        cloth_img_path=cloth_path,
                        mask_img_path=mask_path,
                        pose_json_path=pose_json_path,
                        prompt=prompt,
                        num_inference_steps=steps,
                        strength=0.65
                    )

                    st.image(output_img, caption="Virtual Try-On Result", use_column_width=True)

                except Exception as e:
                    st.error(f" Inference failed: {e}")
