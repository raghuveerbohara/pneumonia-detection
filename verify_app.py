"""Local UI smoke check using a supplied image; does not create cloud evidence."""
import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from inference import load_bundle, predict_upload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    model, metadata, manifest = load_bundle()
    _, result = predict_upload(model, args.image.read_bytes(), args.image.name, metadata)
    upload = io.BytesIO(args.image.read_bytes())
    upload.name = args.image.name

    # Inject an uploaded file because AppTest has no uploader set_value API.
    # The real Predict button and page output are still exercised.
    with patch("streamlit.file_uploader", return_value=upload):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=120)
        app.run()
        assert not app.exception, app.exception
        app.button[0].click().run()
        assert not app.exception, app.exception
        assert not app.error, app.error
        assert len(app.metric) == 2
        assert app.metric[0].value == f"{result['lung_opacity_probability']:.2%}"
        assert app.metric[1].value == f"{result['no_lung_opacity_probability']:.2%}"
        assert app.session_state["result"]["predicted_class"] == result["predicted_class"]

    print("PASS: model load, uploaded-image preprocessing, Predict button and probability display.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
