# ViT-VLM-experiments

I am testing different vision transformers and vision language models with different datasets for different tasks to see their performance

## Project Goal

This notebook is a controlled learning experiment to understand how different pretrained vision backbones and fine-tuning strategies behave under limited compute.

The original goal was not to reproduce the full Food101 benchmark. Instead, the goal was to answer practical ML engineering questions:
- How do pretrained ViT-style models behave when only the classifier head is trained?
- When does it help to unfreeze part of the transformer backbone?
- Is full fine-tuning better than partial fine-tuning?
- Can LoRA provide a useful low-cost adaptation strategy for vision transformers?
- How do transformer-based models compare with CNN-family models like ResNet and ConvNeXt?
- How do confidence, calibration, and error analysis change across fine-tuning strategies?

The final experiment uses a balanced 20-class subset of Food101 with 250 images per class:

```
20 classes × 250 images/class = 5,000 total images
Train split:      4,000 images
Validation split: 1,000 images
```

The validation split was used for epoch-level evaluation, checkpoint selection, model comparison, and final reporting in this notebook.

## Final Dataset Design

The dataset was changed to a balanced 20-class subset:

```python
NUM_SELECTED_CLASSES = 20
SAMPLES_PER_CLASS = 250
TEST_SIZE = 0.20
SEED = 42
```

The final dataset creation logic:
1. Load the Food101 training split.
2. Group image indices by original Food101 class.
3. Select 20 eligible classes reproducibly with a fixed seed.
4. Select exactly 250 images per selected class.
5. Create a stratified train/validation split.
6. Remap original Food101 class IDs to contiguous labels `0..19`.

This fixed the sparsity problem and made per-class metrics meaningful.

## Label Remapping

Original Food101 labels are in the range `0..100`.

After selecting only 20 classes, those labels may not be contiguous. The notebook remaps them:

```
old Food101 class IDs → new labels 0..19
```

This matters because the model classifier head has exactly 20 outputs.

The notebook validates this with class-count checks:

```python
assert set(train_counts.keys()) == set(range(NUM_SELECTED_CLASSES))
assert set(test_counts.keys()) == set(range(NUM_SELECTED_CLASSES))
```

## Preprocessing and Transform Fixes

The notebook uses model-specific image processors from Hugging Face:
- DeiT: `facebook/deit-tiny-patch16-224`
- ResNet-50: `microsoft/resnet-50`
- ConvNeXt-Tiny: `facebook/convnext-tiny-224`

A shared transform function applies the correct processor statistics for each model.

### Training Transform

Training uses randomized augmentation:
```
RandomResizedCrop
ToTensor
Normalize
```

### Validation Transform

Validation uses deterministic preprocessing:
```
Resize
CenterCrop
ToTensor
Normalize
```

This is important because validation should be stable and reproducible.

## Training and Checkpoint Management

The notebook includes a custom `conditional_train_model()` helper that handles:
- Fresh training when `FORCE_FINE_TUNING=True`
- Loading a final saved model when available
- Resuming from the latest checkpoint when available
- Regular Hugging Face model weights
- PEFT/LoRA adapter weights
- Saving final model weights and train metrics

This made the experiment more reliable and reduced the risk of accidentally evaluating a freshly initialized model.

## Evaluation Metrics

The original evaluation mainly used accuracy and F1. This was expanded because the project was specifically investigating confidence and probability separation.

Final metrics include:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall top-1 correctness |
| Macro F1 | Equal-weighted per-class performance |
| Weighted F1 | Support-weighted class performance |
| Top-5 Accuracy | Whether the correct label appears in the top 5 predictions |
| Eval Loss / NLL | Probability assigned to the true class |
| Mean Top-1 Confidence | Average confidence of the predicted class |
| Top-1 / Top-2 Margin | Separation between the best and second-best prediction |
| Correct Mean Confidence | Confidence when predictions are correct |
| Wrong Mean Confidence | Confidence when predictions are wrong |
| ECE | Calibration gap between confidence and empirical accuracy |
| Brier Score | Quality of the full probability distribution |

The notebook also performs qualitative/error analysis:
- Full confusion matrix
- Weakest-class confusion matrix
- Top 10 weakest classes by F1
- Top 10 strongest classes by F1
- Most confident wrong predictions
- Lowest-margin predictions
- Train vs. validation loss curves

## Fine-Tuning Strategies Tested

### 1. Head-Only Fine-Tuning

The backbone is frozen and only the classifier head is trained.

**Purpose:** Test how strong the frozen pretrained representation is and establish a low-cost transfer-learning baseline.

**Final hyperparameters:**
```
Learning rate: 1e-3
Epochs: 10
Effective batch size: 16 × 4 = 64
```

### 2. Full DeiT Fine-Tuning

All DeiT parameters are trainable.

**Purpose:** Test whether full representation adaptation improves performance.

**Differential learning rates were used:**
```python
Backbone LR:   2e-5
Classifier LR: 1e-4
Epochs: 10
Effective batch size: 16 × 4 = 64
```

This is important because the classifier is newly initialized and needs to learn faster, while the pretrained backbone should change more slowly.

### 3. LoRA Fine-Tuning

LoRA was used as a parameter-efficient adaptation method.

**Configuration:**
```
r = 8
alpha = 16
dropout = 0.05
target modules = q_proj, v_proj
learning rate = 2e-4
epochs: 10
effective batch size: 16 × 4 = 64
```

**Purpose:** Test whether a small number of trainable adapter parameters can adapt a pretrained vision transformer.

### 4. Last 3 Transformer Blocks + Head

All layers are frozen except:
```
- Last 3 transformer encoder blocks
- Classifier head
```

**Purpose:** Test whether moderate representation adaptation is enough.

### 5. Last 5 Transformer Blocks + Head

All layers are frozen except:
```
- Last 5 transformer encoder blocks
- Classifier head
```

**Purpose:** Test whether a slightly deeper partial adaptation improves over last-3-block fine-tuning.

### 6. ResNet-50 Full Fine-Tuning

A classic CNN baseline was added using `microsoft/resnet-50`.

**Purpose:** Compare the transformer models against a standard convolutional baseline.

### 7. ConvNeXt-Tiny Full Fine-Tuning

A modern CNN-family baseline was added using `facebook/convnext-tiny-224`.

**Purpose:** Compare DeiT not only against classic CNNs, but also against a modern CNN architecture influenced by transformer-era design choices.

## Why the Model Choice Changed

The first ViT experiments used a larger model. Under limited compute, the results were weak and unstable. Partial fine-tuning worked better than full fine-tuning, but the overall accuracy was low.

The model was changed to `facebook/deit-tiny-patch16-224`.

This significantly improved the experiment because DeiT-Tiny is smaller, faster, and easier to fine-tune with limited data and compute. This was a major turning point in the project.

## Final Results

### DeiT / ResNet Comparison

| Strategy | Trainable Params (M) | Epochs | LR | Accuracy | Macro F1 | Top-5 Acc | Eval Loss | Mean Confidence | Mean Margin | ECE | NLL | Brier |
|----------|----------------------|--------|-----|----------|----------|-----------|-----------|-----------------|-------------|-----|-----|-------|
| ViT Last 5 Blocks + Head | 2.23 | 10 | 1e-4 | 0.833 | 0.8330 | 0.970 | 0.5703 | 0.8108 | 0.7377 | 0.0345 | 0.5703 | 0.2332 |
| ViT Last 3 Blocks + Head | 1.34 | 10 | 1e-4 | 0.816 | 0.8158 | 0.961 | 0.6534 | 0.7582 | 0.6671 | 0.0616 | 0.6534 | 0.2625 |
| ViT Full Fine-tune | 5.53 | 10 | differential | 0.793 | 0.7929 | 0.957 | 0.7729 | 0.6746 | 0.5650 | 0.1184 | 0.7729 | 0.3088 |
| ViT LoRA Fine-tune | 0.08 | 10 | 2e-4 | 0.754 | 0.7506 | 0.951 | 0.8922 | 0.6289 | 0.4946 | 0.1274 | 0.8922 | 0.3685 |
| ViT Head-Only Fine-tune | ~0.00 | 10 | 1e-3 | 0.740 | 0.7366 | 0.947 | 0.9417 | 0.6137 | 0.4802 | 0.1263 | 0.9417 | 0.3926 |
| ResNet-50 Full Fine-tune | 23.55 | 10 | 1e-4 | 0.703 | 0.6990 | 0.933 | 1.3063 | 0.4258 | 0.3061 | 0.2772 | 1.3063 | 0.5167 |

### ConvNeXt-Tiny Result

ConvNeXt-Tiny outperformed all previously tested models.

**Best validation result:**

| Model | Trainable Params (M) | Epochs | LR | Accuracy | Macro F1 | Weighted F1 | Top-5 Acc | Eval Loss | Mean Confidence | Mean Margin | ECE | NLL | Brier |
|-------|----------------------|--------|-----|----------|----------|-------------|-----------|-----------|-----------------|-------------|-----|-----|-------|
| ConvNeXt-Tiny Full Fine-tune | 27.84 | 10 | 1e-4 | 0.879 | 0.8792 | 0.8792 | 0.981 | 0.4439 | 0.8150 | 0.7458 | 0.0642 | 0.4439 | 0.1800 |

The training log showed the model continued improving over epochs and reached its best validation accuracy around epoch 8:

```
Epoch 8 accuracy: 0.879
Epoch 8 macro F1: 0.8792
Epoch 8 top-5 accuracy: 0.981
Epoch 8 validation loss: 0.4439
```

The final epoch remained close:

```
Epoch 10 accuracy: 0.876
Epoch 10 macro F1: 0.8760
Epoch 10 top-5 accuracy: 0.984
Epoch 10 validation loss: 0.4254
```

## Interpretation of Results

### 1. ConvNeXt-Tiny performed best

ConvNeXt-Tiny reached the strongest overall validation performance:
```
Accuracy: 87.9%
Macro F1: 87.9%
Top-5 Accuracy: 98.1%
Eval Loss: 0.4439
```

This suggests that the Food101 subset benefits from both:
- Local texture recognition
- Broader spatial/contextual structure

ConvNeXt is well suited to this because it keeps the convolutional inductive bias of CNNs while using modern design choices such as larger kernels, LayerNorm, GELU, depthwise convolutions, and transformer-inspired architectural elements.

### 2. Partial DeiT fine-tuning was the best transformer strategy

Among the DeiT strategies, unfreezing the last 5 transformer blocks plus the classifier head worked best.

This suggests:
- Frozen DeiT features were already useful
- Head-only training was a strong baseline
- Adapting later transformer layers improved class separation
- Full fine-tuning was not necessary and was slightly less effective under this data/compute budget

### 3. Head-only fine-tuning was surprisingly strong

Head-only fine-tuning reached:
```
Accuracy: 74.0%
Top-5 Accuracy: 94.7%
```

This confirms that the pretrained DeiT representation already transfers well to the selected Food101 classes.

### 4. LoRA was parameter-efficient but not the best performing method

LoRA trained only about 0.08M parameters and reached:
```
Accuracy: 75.4%
Top-5 Accuracy: 95.1%
```

This is close to head-only and much cheaper than full fine-tuning, but it did not match partial block unfreezing.

**Interpretation:**
- LoRA provided some adaptation
- The task benefited more from directly adapting later transformer blocks
- LoRA remains useful when memory/storage constraints are strict

### 5. ResNet-50 underperformed and appeared underconfident

ResNet-50 reached:
```
Accuracy: 70.3%
Macro F1: 69.9%
Top-5 Accuracy: 93.3%
```

This is not a failure, but its probability quality was weaker:
```
ECE: 0.2772
NLL: 1.3063
Mean confidence: 0.4258
Mean margin: 0.3061
```

The model was often correct, but less confident than the DeiT and ConvNeXt models. This points to underconfidence and weaker calibration.

The training loss was also noticeably high compared with validation loss. This may be due to:
- Less suitable optimization settings
- Architecture/pretraining mismatch
- ResNet being a classic CNN baseline rather than a modern CNN design

ConvNeXt's stronger result shows that CNN-family models were not inherently unsuitable for the task; rather, ResNet-50 was less effective under this setup than ConvNeXt-Tiny.

## Calibration and Confidence Findings

The project initially focused on the observation that some models predicted the right class but with very close probabilities. The added metrics made this measurable.

**Important findings:**
- Better models had higher top-1/top-2 margins
- Correct predictions had higher confidence than wrong predictions
- ConvNeXt and the best DeiT models had reasonable ECE
- ResNet-50 had much higher ECE and NLL, suggesting poorer probability calibration
- Top-5 accuracy was high across most models, meaning many models often included the correct class among their top candidates even when top-1 ranking failed

**For example:**
```
ConvNeXt-Tiny:
  Accuracy = 0.879
  Mean Confidence = 0.815
  Mean Margin = 0.7458
  ECE = 0.0642

DeiT Last 5 Blocks:
  Accuracy = 0.833
  Mean Confidence = 0.8108
  Mean Margin = 0.7377
  ECE = 0.0345

ResNet-50:
  Accuracy = 0.703
  Mean Confidence = 0.4258
  Mean Margin = 0.3061
  ECE = 0.2772
```

This supports the conclusion that ConvNeXt and partially fine-tuned DeiT learned stronger class separation.

## Key Lessons Learned

### Dataset design matters

A balanced 20-class subset made the experiment meaningful and easier to interpret.

### Model size matters

The larger ViT was harder to fine-tune under limited compute. DeiT-Tiny worked much better because it was smaller and more suitable for the resource budget.

### Fine-tuning strategy matters

Full fine-tuning was not always best. Partial unfreezing of later layers was better for DeiT.

### Learning rates should differ by strategy

The final notebook uses different learning rates for different strategies:
- Head-only: higher LR
- Full fine-tuning: lower LR for backbone, higher LR for classifier
- LoRA: higher adapter LR
- Partial unfreezing: moderate LR

This made training much healthier.

### CNN vs ViT is not a simple comparison

ResNet-50 underperformed, but ConvNeXt-Tiny performed best. This shows that "CNN vs ViT" is too broad. Architecture details and training dynamics matter.

### Evaluation should go beyond accuracy

Accuracy alone would have missed important findings about:
- Confidence
- Probability margins
- Calibration
- Top-5 behavior
- Per-class weaknesses

## Limitations

This experiment has several important limitations:

**Validation set used for model selection:** The reported metrics are validation results. A separate held-out test split would be needed for an unbiased final benchmark.

**Subset of Food101 only:** The results apply to the selected 20-class subset, not full Food101.

**Random class selection:** The selected 20 classes may affect results. Easier or more visually distinct classes may inflate performance.

**Limited hyperparameter search:** Learning rates and epochs were tuned manually, not through a full systematic sweep.

**Compute-limited training:** The models were selected and tuned under limited resources. Larger models may perform better with more compute and data.

## Final Summary

This project evolved from a simple ViT fine-tuning notebook into a controlled comparison of vision backbones and fine-tuning strategies.

**The major improvements were:**
- A balanced 20-class subset
- Remapping labels correctly to a 20-class classifier head
- Improved transform and data collator issues
- Expanding evaluation beyond accuracy
- Tuning learning rates per strategy
- Comparing transformer, classic CNN, and modern CNN architectures

**The best model was ConvNeXt-Tiny, followed by partial DeiT fine-tuning.** The results suggest that Food101 classification benefits from both local texture understanding and broader image context, and that modern CNN designs can compete with or outperform vision transformers under appropriate conditions.
