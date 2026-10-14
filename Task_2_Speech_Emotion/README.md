# Task 2: Speech Emotion Recognition
Recognizes 8 RAVDESS acted emotions using 40 MFCC coefficients and a **PyTorch 1-D CNN**. Train/test split is **actor-independent** to reduce speaker leakage. Outputs classification report, held-out accuracy, confusion matrix and trained weights.

## Required external dataset
Download **RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**, specifically **Audio_Speech_Actors_01-24.zip**, from the official Zenodo record: https://zenodo.org/records/1188976 . Extract the ZIP contents (folders `Actor_01`, ..., `Actor_24`) under `Task_2_Speech_Emotion/data/raw/`. Audio files are not redistributed in this project; review dataset licensing and citation. Folder nesting is fine.

Run `python Task_2_Speech_Emotion/train.py`. The notebook will also work when opened inside its task folder. Internet is not required **after** downloading the audio. The system deliberately raises a helpful error when there are no real audio labels — it does not invent a score.

Limitations: emotion classes are acted in a controlled dataset; real conversations, languages, noisy environments and diverse speakers may perform differently. A single train/test split provides a baseline, not deployment-grade validation.
