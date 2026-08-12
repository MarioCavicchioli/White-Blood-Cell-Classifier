# WBC Classifier — Notebook Fix Spec

**File:** `submission.ipynb` (WBCBench2026 submission)
**Goal:** The notebook currently trains to ~1.5% validation accuracy (worse than the ~7.7% you'd get from random guessing on 13 classes) and then crashes after epoch 1. Two bugs cause this. Fix them to get back to the working ~0.7 macro-F1 baseline, then apply the cleanups.

---

## Critical bug 1 — Learning rate is ~100× too high

**Where:** the optimizer cell (currently the cell that defines `loss_fn`, `optimizer`, `scheduler`).

**Current:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
```

**Problem:** `lr=0.1` with Adam is enormous for fine-tuning a pretrained ResNet50. The first few updates destroy the pretrained weights, so training loss climbs instead of falling and the model never learns. This is the main reason every recent run was "rubbish."

**Fix:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

---

## Critical bug 2 — Scheduler `T_max=0` crashes training

**Where:** same cell.

**Current:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0)
```

**Problem:** `T_max=0` is invalid. It throws `ZeroDivisionError: integer modulo by zero` when `scheduler.step()` runs at the end of epoch 1, which is why the run dies after a single epoch. `T_max` should be the number of epochs the learning rate is annealed over.

**Fix:** set `T_max` to the number of training epochs. Note an ordering issue: `EPOCHS` is currently defined later in the training-loop cell, *after* the scheduler is created, so referencing `EPOCHS` here would raise a `NameError`. Move the `EPOCHS` definition up so it is defined **before** the scheduler.

```python
EPOCHS = 10                       # define here, before the scheduler
# ...
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

Then remove the now-duplicate `EPOCHS = 10` line from the training-loop cell.

---

## Cleanup 1 — Class imbalance is being corrected twice

The notebook compensates for imbalance in two places at once:

1. `WeightedRandomSampler` in the data-loader cell (oversamples minority classes into each batch), and
2. `class_weights` passed to `CrossEntropyLoss` (up-weights minority classes in the loss).

Doing both double-counts and can over-correct rare classes (and the sampler with `replacement=True` repeats the same rare images a lot, risking overfitting). **Keep one.** Recommendation: keep the weighted loss, drop the sampler.

**In the data-loader cell**, remove the sampler and shuffle instead:
```python
# delete the sample_weights / sampler lines, then:
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```
Leave `val_loader` and `test_loader` as they are (`shuffle=False`).

---

## Cleanup 2 — Remove the unused FocalLoss

A `FocalLoss` class is defined but never used — training runs on `CrossEntropyLoss`. Delete the `FocalLoss` class for now so the notebook doesn't imply it's doing something it isn't. (It can be reintroduced later as an experiment — see below — but pick one loss and use it.)

---

## Cleanup 3 (minor) — Modernise the deprecated weights arg

**Current:**
```python
self.model = models.resnet50(pretrained=True)
```
**Fix (silences the deprecation warning):**
```python
from torchvision.models import ResNet50_Weights
self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
```

---

## After fixing — how to check it worked

Rerun the training cell. You should see, within the first few epochs:
- Training loss **falling** (not stuck around 10),
- Validation accuracy climbing well past 7.7%,
- Validation macro-F1 climbing toward ~0.7.

The competition is scored on **macro-F1**, so that's the number to watch — not raw accuracy.

---

## Optional improvements to try *after* the baseline is back

Only once it's training to ~0.7 again — change one thing at a time so you can attribute any gain:
- **Differential learning rates:** lower LR for the unfrozen backbone (`layer2/3/4`, e.g. `1e-5`) and a higher LR for the new `fc` head (e.g. `1e-3`), via optimizer parameter groups.
- **Train a bit longer** (e.g. 15–20 epochs) now that the scheduler works, watching for overfitting.
- **Swap in FocalLoss** (gamma ≈ 2) in place of weighted CrossEntropyLoss and compare macro-F1 — this is where the FocalLoss class earns its place.
