"""Streamlit frontend with shared Python inference backend."""
import json
import os
from datetime import datetime, timezone

import streamlit as st

from inference import load_bundle, predict_upload

st.set_page_config(page_title="Chest X-ray Opacity Classifier", layout="centered")
st.title("Pneumonia Detection — Capstone Demo")
st.write("Upload a chest X-ray to predict the supplied lung-opacity class.")
st.caption("Educational model demonstration. This model does not confirm or exclude pneumonia.")


@st.cache_resource(show_spinner="Loading the selected model…")
def cached_bundle():
    return load_bundle()


uploaded = st.file_uploader("Upload a chest X-ray", type=["dcm", "png", "jpg", "jpeg"])

if uploaded is not None:
    if st.button("Predict", type="primary"):
        try:
            with st.spinner("Processing the X-ray…"):
                model, metadata, manifest = cached_bundle()
                image, result = predict_upload(model, uploaded.getvalue(), uploaded.name, metadata)
                result["model_sha256"] = manifest["model_sha256"]
                result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
                result["runtime"] = {
                    "codespace_name": os.environ.get("CODESPACE_NAME"),
                    "codespaces": os.environ.get("CODESPACES") == "true",
                }
                st.session_state["result"] = result
                st.session_state["image"] = image
                st.session_state["input_bytes"] = uploaded.getvalue()
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Prediction could not be completed. Check the model files and application logs.")

    # Only show a result for the file that was actually predicted.
    if st.session_state.get("input_bytes") == uploaded.getvalue():
        result = st.session_state["result"]
        st.image(st.session_state["image"][..., 0], caption="Preprocessed chest X-ray", clamp=True)
        st.subheader("Prediction")
        st.write(f"**Predicted class:** {result['predicted_class']}")
        left, right = st.columns(2)
        left.metric("Lung-opacity probability", f"{result['lung_opacity_probability']:.2%}")
        right.metric("No-lung-opacity probability", f"{result['no_lung_opacity_probability']:.2%}")
        st.caption("These are uncalibrated model probabilities. No lung opacity does not mean a normal X-ray.")
        st.caption(f"Decision threshold: {result['threshold']:.6f}")
        st.download_button(
            "Download prediction record", json.dumps(result, indent=2),
            file_name="prediction_record.json", mime="application/json",
        )

with st.expander("About this model"):
    st.write("Frozen DenseNet121 with a regularized three-class classifier; the opacity score is used for the binary prediction.")
    st.write("The decision threshold was selected using validation data and is fixed in this app.")
    st.write("PNG/JPEG images follow the same resizing and normalization sequence, but changing image format can change predictions compared with the original DICOM.")
