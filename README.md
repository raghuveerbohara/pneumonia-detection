# Pneumonia Detection — Streamlit Deployment

Educational chest X-ray **lung-opacity** classifier. The selected model is
DenseNet121_regularized3. Model predictions are not confirmed pneumonia diagnoses.

## Architecture and dependencies

- Frontend: Streamlit uploader, image preview, class, probabilities and JSON download.
- Backend: `inference.py` with TensorFlow/Keras, NumPy, Pillow and pydicom.
- A separate Flask API is not required; Streamlit calls the Python inference module.
- Complete pinned runtime requirements are in `requirements.txt`.
- Docker: Python 3.11, TensorFlow CPU 2.20.0, bundled trained model, port 8501.

Uploads are processed in the active session and are not written to the repository.
Supported formats: single-frame uncompressed DICOM, PNG and JPEG, up to 20 MiB.
Compressed DICOM files requiring extra pixel decoders may need conversion first.

## Model files

The exact trained model is split into browser-upload-sized parts in `models/`.
`restore_model.py` verifies part hashes, reconstructs `best_model.keras`, and checks
its complete SHA-256. The split changes storage only, not model weights.
The restored file is gitignored because the parts already contain the model.
The Docker image reconstructs and **includes the model during image building**.

`best_model_metadata.json` stores the validation-selected threshold. Do not replace
it with 0.50. Inputs are grayscale, float32 [0,1], shape (224,224,1). The shared
preprocessor matches the notebook for supported DICOM files. Pretrained-network
scaling, feature normalization and classifier layers are inside the Keras model.

## Run locally

From this directory, using a Python environment with the listed dependencies:

```bash
python restore_model.py
python -m streamlit run app.py
```

On Windows, the existing notebook's Anaconda environment can run the app without
reinstalling TensorFlow. The Docker requirements use tensorflow-cpu for Linux.

## Build and run Docker

```bash
docker build -t pneumonia-detection .
docker run --rm -p 8501:8501 pneumonia-detection
```

Open http://localhost:8501 and upload a sample chest X-ray.

## Create the GitHub repository and upload

1. Create an empty repository in your GitHub account, e.g. `pneumonia-detection`.
2. Upload the **contents** of this directory as the repository root. Include
   `models/`, `.streamlit/` and `.devcontainer/`. Do not upload the dataset,
   feature caches, assignment PDFs or your entire capstone directory.
3. The model pieces are below GitHub's browser-upload file limit. Commit the files.
4. Confirm the repository root contains `Dockerfile`, `app.py` and `.devcontainer/`.

If using Git locally instead, initialize and push **this directory only**:

```bash
git init
git add .
git commit -m "Add Streamlit model deployment and Docker configuration"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection.git
git push -u origin main
```

Authenticate using GitHub's normal login flow. Never place tokens in notebook cells.

## Run in GitHub Codespaces

1. From the repository, choose **Code → Codespaces → Create codespace on main**.
2. The dev container builds using the same Dockerfile as the portable app image.
   The first build downloads dependencies and can take several minutes.
3. In the Codespaces terminal, run:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

4. In the **Ports** panel, open forwarded port **8501** in the browser.
5. Record the actual **Forwarded Address**. It typically resembles
   `https://YOUR-CODESPACE-NAME-8501.app.github.dev`; do not submit this placeholder.
6. Upload a known chest X-ray, press **Predict**, and download the prediction JSON.
7. Capture the running app with its URL, class and probabilities, and the Ports panel.
8. Save the repository URL, forwarded URL, screenshots and prediction JSON for
   the report and executed notebook. The app adds Codespaces runtime information
   to records generated there; local inference does not count as Codespaces evidence.

Port access is private by default. If evaluator access is required, choose the
appropriate visibility through GitHub's Ports panel. A Codespaces URL works only
while that Codespace is running; it is a demonstration URL, not permanent hosting.

## Sources

- https://docs.streamlit.io/deploy/tutorials/docker
- https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Model architecture/weights originate from Keras Applications; training and
  evaluation are documented in the project's final notebook.
