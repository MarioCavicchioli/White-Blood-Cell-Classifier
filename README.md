White Blood Cell Classifier

A deep-learning image classifier that sorts microscope images of white blood cells into 13 types, built in PyTorch for the WBCBench2026 competition.

The interesting part of this problem isn't the model — it's the imbalance. Some cell types appear thousands of times in the data and others only a handful, so a model can score high accuracy by simply ignoring the rare classes. This project is built around measuring and handling that honestly.

Results
Metric	Validation
Macro-F1	~0.70
Accuracy	~89%

The gap between those two numbers is the whole story. Accuracy (~89%) looks great, but it's flattered by the common cell types. Macro-F1 (~0.70) weights all 13 classes equally, so it reflects how the model does on the rare cells too — which is why the competition is scored on it, and why it's the number I actually pay attention to.

The data
~30,000 labelled microscope images across train / validation / test splits
13 classes of white blood cell, heavily imbalanced (common types appear thousands of times; the rarest only a few dozen)
Images vary in size, so all are resized to 224×224 for the model
Approach
Transfer learning with a ResNet50 pretrained on ImageNet — early layers frozen, deeper blocks (layer2–layer4) fine-tuned, and a new classifier head (dropout + linear) trained from scratch.
Class imbalance handled with a class-weighted cross-entropy loss, so rare cell types aren't drowned out by common ones.
Data augmentation (horizontal/vertical flips, small rotations) to make the model more robust, since a blood cell has no fixed orientation.
Training: Adam (lr 1e-4, weight decay 1e-4) with a cosine-annealing learning-rate schedule over 10 epochs, on GPU (CUDA).
Evaluation: macro-F1 tracked every epoch on a held-out validation split, alongside accuracy and loss.
Running it

Requirements:

bash
pip install torch torchvision scikit-learn pandas pillow tqdm
Download the WBCBench2026 dataset and set BASE_DIR at the top of the notebook to point at it.
Open and run submission.ipynb top to bottom.
It trains the model, reports validation macro-F1 per epoch, and writes submission.csv (the test-set predictions).

A CUDA-capable GPU is strongly recommended — training on CPU is very slow.

Repo contents
submission.ipynb — the full pipeline: data loading, model, training loop, evaluation, and prediction export.
What I'd try next

The current model is a solid baseline. Honest next steps, changing one thing at a time so any gain is attributable:

Differential learning rates — a lower rate for the pretrained backbone and a higher one for the new head.
Best-checkpoint saving — keep the epoch with the highest validation macro-F1 rather than the last one.
Focal loss in place of weighted cross-entropy, to lean harder into the rare classes.
Targeting the rarest classes specifically, which are what currently cap the macro-F1.
