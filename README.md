# CodeAlpha | Machine Learning Internship Projects

**Four independent tasks** completed as reproducible Python pipelines, with separate notebooks and README files. Prepared for GitHub.

| Folder | Project | Data source | Method |
|---|---|---|---|
| Task_1_Credit_Scoring | Credit scoring | Synthetic demonstration by default; optional real CSV | Random Forest / Logistic Regression |
| Task_2_Speech_Emotion | Emotion recognition | RAVDESS audio (download separately) | MFCC + 1-D CNN |
| Task_3_Handwriting | Handwritten digits | Built-in sklearn digits (8×8, offline) | 2-D CNN |
| Task_4_Disease_Prediction | Breast cancer classification | Built-in sklearn Wisconsin breast cancer | Logistic Regression / Random Forest |

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a task, e.g. `python Task_1_Credit_Scoring/train.py`. Open `jupyter notebook` for notebooks. Run Task 2 *after downloading RAVDESS audio* (see its README). Trained model weights and images are created under `outputs/` and intentionally excluded from Git history.

## Reproducibility and limitations

Task 1 defaults to **synthetic, non-real credit data**, so results are demonstration-only. Replace with consented real records and assess fairness before any real-world use. Task 2 cannot produce valid emotion metrics without labeled real audio. Task 3 uses sklearn digits rather than the PDF's suggested MNIST/EMNIST: it recognizes **digits**, not letters or handwriting at full resolution. Task 4 is a **technical ML example, not a medical diagnosis tool**. Dataset split and evaluation are reproducible.

## Submission

CodeAlpha requests 2 or 3 out of 4 tasks (minimum 2), a GitHub repository named `CodeAlpha_ProjectName`, a LinkedIn project video with GitHub link, and submission through the form from the WhatsApp group. These four directories can be published together, but if an individual-project repo is required, create separate repos (e.g. `CodeAlpha_CreditScoring`, `CodeAlpha_SpeechEmotion`, etc.).

To upload from GitHub website, create a public repository, choose **Add file → Upload files**, and drag the project folders and root files onto the upload area. Do not upload the ZIP as the only file. Exclude the external RAVDESS WAV dataset.
