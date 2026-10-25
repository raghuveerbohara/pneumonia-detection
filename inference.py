"""Shared model loading, preprocessing and inference for UI and verification."""
import io
import json
import hashlib
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageOps, UnidentifiedImageError

from restore_model import ROOT, restore_model

IMAGE_SIZE = (224, 224)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_777_216


def normalize_grayscale(grayscale):
    grayscale = np.asarray(grayscale, dtype=np.float32)
    if grayscale.ndim != 2 or not np.isfinite(grayscale).all():
        raise ValueError("The image must contain a single finite grayscale pixel array.")
    if grayscale.size > MAX_PIXELS:
        raise ValueError("The image is too large. Use an image no larger than 4096 × 4096 pixels.")
    if grayscale.max() <= grayscale.min():
        raise ValueError("The image has no intensity variation; upload a valid chest X-ray.")
    # Exactly the float-mode resize/min-max sequence used in the notebook.
    height, width = IMAGE_SIZE
    resized = np.asarray(
        Image.fromarray(grayscale).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    low, high = resized.min(), resized.max()
    normalized = (resized-low)/(high-low) if high > low else np.zeros_like(resized)
    return np.clip(normalized, 0, 1).astype(np.float32)[..., np.newaxis]


def preprocess_upload(content, filename):
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Upload a nonempty file smaller than 20 MiB.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".dcm":
        try:
            dataset = pydicom.dcmread(io.BytesIO(content))
            if int(dataset.get("NumberOfFrames", 1)) != 1:
                raise ValueError("Multi-frame DICOM files are not supported.")
            pixels = dataset.pixel_array
            photometric = dataset.get("PhotometricInterpretation")
            if pixels.ndim == 2 and photometric in {"MONOCHROME1", "MONOCHROME2"}:
                grayscale = pixels.astype(np.float32)
            elif pixels.ndim == 3 and pixels.shape[-1] == 3 and photometric == "RGB":
                rgb = pixels.astype(np.float32)
                grayscale = .299*rgb[..., 0] + .587*rgb[..., 1] + .114*rgb[..., 2]
            else:
                raise ValueError("Use a single grayscale or RGB DICOM image.")
            if photometric == "MONOCHROME1":
                grayscale = grayscale.max()+grayscale.min()-grayscale
        except (ValueError, NotImplementedError, RuntimeError) as error:
            raise ValueError(f"Cannot read DICOM image: {error}") from error
        except Exception as error:
            raise ValueError("Cannot read this DICOM file. Use an uncompressed DICOM or PNG/JPEG image.") from error
    elif suffix in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(content)) as original:
                if original.format not in {"PNG", "JPEG"}:
                    raise ValueError("The file content must be PNG or JPEG.")
                if original.width * original.height > MAX_PIXELS:
                    raise ValueError("The image exceeds the supported pixel count.")
                image = ImageOps.exif_transpose(original)
                if image.mode in {"L", "I", "F", "I;16", "I;16B", "I;16L"}:
                    grayscale = np.asarray(image, dtype=np.float32)
                else:
                    if image.mode == "RGBA" or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        background = Image.new("RGBA", rgba.size, "white")
                        image = Image.alpha_composite(background, rgba)
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                    grayscale = .299*rgb[..., 0] + .587*rgb[..., 1] + .114*rgb[..., 2]
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Cannot read the uploaded PNG/JPEG image.") from error
    else:
        raise ValueError("Supported formats are .dcm, .png, .jpg and .jpeg.")
    return normalize_grayscale(grayscale)


def load_bundle():
    import tensorflow as tf
    model_path, manifest = restore_model()
    metadata = json.loads((ROOT / "models" / "best_model_metadata.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape != (None, 224, 224, 1) or model.output_shape != (None, 1):
        raise ValueError("Unexpected model input/output contract.")
    if model.name != metadata["model"]:
        raise ValueError("Model identity and threshold metadata do not match.")
    threshold = float(metadata["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid saved decision threshold.")
    return model, metadata, manifest


def predict_array(model, image, metadata):
    if image.shape != (224, 224, 1) or not np.isfinite(image).all():
        raise ValueError("Invalid preprocessed image.")
    score = float(model(image[np.newaxis], training=False).numpy().ravel()[0])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid probability score.")
    predicted = int(score >= float(metadata["threshold"]))
    return {
        "model": metadata["model"],
        "threshold": float(metadata["threshold"]),
        "predicted_target": predicted,
        "predicted_class": "Lung opacity" if predicted else "No lung opacity",
        "lung_opacity_probability": score,
        "no_lung_opacity_probability": 1-score,
        "probability_note": "Uncalibrated model score for the supplied opacity label; not a clinical pneumonia probability.",
    }


def predict_upload(model, content, filename, metadata):
    image = preprocess_upload(content, filename)
    result = predict_array(model, image, metadata)
    result["input_sha256"] = hashlib.sha256(content).hexdigest()
    return image, result
