# Task 3: Handwritten Character Recognition (digits 0–9)
A genuine PyTorch **2D CNN**, trained on the scikit-learn handwritten digits dataset. It performs stratified train/validation/test splits, chooses a checkpoint using only validation results, then prints final test accuracy and a confusion matrix.

Run `python Task_3_Handwriting/train.py`. Data is **8×8 digits**, not 28×28 MNIST and not EMNIST alphabet characters. This is a fully offline digit-recognition implementation of the task's CNN approach. Extending to letters requires an EMNIST dataset and a different output layer.
