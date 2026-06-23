# Classifier vs VLM and Classifier+VLM Hybrid Experiments

## Overview

This project explored whether a fine-tuned image classifier, a zero-shot vision-language model (VLM), or a hybrid classifier+VLM pipeline is the best approach for a fixed-label food image classification task.

The task used a balanced 20-class subset of Food101. The main comparison was between:

1. A fine-tuned ConvNeXt-Tiny image classifier.
2. A zero-shot VLM used as a food classifier.
3. A hybrid pipeline where ConvNeXt predicts first, and the VLM is only used to revalidate uncertain ConvNeXt predictions.


## Dataset Setup
---

The experiment used a 20-class subset of Food101 instead of the full 101-class dataset.

The validation/test-style split contained 1,000 images, with 50 images per class.

This setup allowed the experiments to compare closed-set classifier performance, zero-shot VLM behavior, top-k candidate quality, and whether a VLM can improve classifier mistakes when given a candidate list.


## Experiment 1: Fine-Tuned Classifier Baseline
---

The first main baseline was a fine-tuned ConvNeXt-Tiny model.

ConvNeXt was chosen because it is a strong modern CNN-style model. Compared with earlier ViT/DeiT attempts, ConvNeXt gave stronger accuracy and more reliable performance on this limited-data setup.

### ConvNeXt results

| Metric | Result |
|---|---:|
| Accuracy | 0.8850 |
| Macro F1 | 0.8856 |
| Weighted F1 | 0.8856 |
| Top-5 Accuracy | 0.9810 |
| Eval Loss | 0.4299 |
| ECE | 0.0689 |
| NLL | 0.4299 |
| Brier Score | 0.1755 |
| Training Time | 1035.60 sec |


## Experiment 2: Zero-Shot VLM Single-Output Classification
---

The next experiment tested a VLM as a zero-shot classifier. The VLM received the image and a prompt containing the full 20-class label list. It was asked to output exactly one class name.

This tested whether the VLM could classify food images without task-specific fine-tuning.

### VLM single-output results

| Metric | Result |
|---|---:|
| Accuracy | 0.8060 |
| Macro F1 | 0.8101 |
| Weighted F1 | 0.8101 |
| Invalid Output Rate | 0.0030 |
| Evaluation Time | 1565.16 sec |
| Avg Time per Image | 1.57 sec/image |

### Interpretation

The zero-shot VLM was surprisingly competitive. It reached about 80.6% accuracy without fine-tuning, which shows that the VLM understood the visual task and followed the class-list prompt well.

However, it was much slower than ConvNeXt. For a fixed-label classification task, the VLM is not a good replacement for the fine-tuned classifier.

The conclusion from this stage was:

```text
ConvNeXt is faster and more accurate for fixed-label classification.
The VLM is flexible and works zero-shot, but it is too slow as the primary classifier.
```


## Experiment 3: Zero-Shot VLM Forced-Choice Scoring
---

The next VLM experiment used forced-choice scoring. Instead of asking the VLM to generate one answer, each candidate label was scored separately.

The idea was:

```text
image + prompt + candidate label -> score
```

This produces one score per class, creating a `[num_images, num_classes]` score matrix similar to classifier logits.

### Why this was useful

Forced-choice scoring allowed probability-style comparisons such as top-1 prediction, top-5 prediction, softmax over class scores, confidence, and calibration-style metrics.

### Why it was impractical

This method was much slower because each image required scoring every candidate label.

For 1,000 images and 20 labels:

```text
1000 images x 20 candidate labels = 20,000 image-text scoring examples
```

This was too expensive for the Colab setup. The forced-choice run started correctly but was not practical as a full evaluation path.

### Interpretation

Forced-choice VLM scoring is useful for understanding how VLMs can be turned into classifiers, but it is not efficient enough for this task in limited compute.


## Experiment 4: Initial Classifier+VLM Diagnostic Hybrid
---

After comparing ConvNeXt and the VLM separately, the next question was whether the VLM could add value as a second-stage reviewer.

The diagnostic setup selected:

1. low-confidence correct ConvNeXt predictions,
2. all incorrect ConvNeXt predictions.

This was not fully realistic because, at inference time, the system would not know which predictions are incorrect. However, it was useful for testing whether the VLM could fix ConvNeXt mistakes when given the right candidate list.

### Diagnostic hybrid results

| Metric | Result |
|---|---:|
| ConvNeXt Accuracy | 0.8790 |
| ConvNeXt Top-5 Accuracy | 0.9840 |
| Low-Confidence Correct Cases | 73 |
| Incorrect Cases | 121 |
| Total Routed Cases | 194 |
| Incorrect True Label Coverage in Candidates | 0.9587 |

### VLM revalidation results

| Metric | Result |
|---|---:|
| Total Selected Cases | 194 |
| Low-Confidence Correct Cases | 73 |
| Incorrect Cases | 121 |
| VLM Invalid Outputs | 0 |
| Correct Cases Preserved by VLM | 48 |
| Correct Cases Regressed by VLM | 25 |
| Incorrect Cases Fixed by VLM | 81 |
| Incorrect Cases Same Wrong by VLM | 7 |
| Incorrect Cases Different Wrong by VLM | 33 |
| Net Change on Selected Cases | +56 |
| Avg Latency per Selected Sample | 1.28 sec |

### Interpretation

This showed that the VLM could fix many ConvNeXt mistakes when the true label was present in the candidate list. However, it also regressed some correct predictions.


## Experiment 6: Realistic Inference-Time Classifier+VLM Routing
---

The final experiment removed the oracle component.

Instead of selecting known incorrect predictions, the system routed samples based only on information available at inference time:

- ConvNeXt top-1 confidence,
- ConvNeXt top1-top2 margin,
- optionally entropy.

The VLM was then given the original image plus ConvNeXt's top candidate labels and asked to choose the best class.

This is the realistic hybrid setup:

```text
image
  -> ConvNeXt classifier
  -> if uncertain, send image + top-k candidates to VLM
  -> otherwise, trust ConvNeXt
```

### Final hybrid results

| Metric | Result |
|---|---:|
| Total Dataset Samples | 1000 |
| ConvNeXt Correct Total | 875 |
| ConvNeXt Accuracy | 0.8750 |
| Routed Uncertain Cases | 174 |
| Routing Rate | 0.1740 |
| Routed Originally Correct Cases | 80 |
| Routed Originally Wrong Cases | 94 |
| VLM Invalid Outputs | 0 |
| Originally Correct Preserved by VLM | 52 |
| Originally Correct Regressed by VLM | 28 |
| Originally Wrong Fixed by VLM | 66 |
| Originally Wrong Same Wrong by VLM | 1 |
| Originally Wrong Different Wrong by VLM | 27 |
| True Label Candidate Coverage | 1.0000 |
| Net Change on Routed Cases | +38 |
| Hybrid Correct Total | 913 |
| Hybrid Accuracy | 0.9130 |
| Hybrid Accuracy Gain | +0.0380 |
| VLM Revalidation Time | 226.40 sec |
| Avg Latency per Routed Sample | 1.30 sec |
| Avg Added Latency per Dataset Sample | 0.226 sec |

### Interpretation

This was the most important experiment.

The ConvNeXt-only model achieved 87.5% accuracy. By routing only 17.4% of samples to the VLM, the hybrid system improved accuracy to 91.3%.

The VLM fixed 66 ConvNeXt mistakes but regressed 28 originally correct predictions, resulting in a net gain of 38 samples.

This shows that the VLM adds value when used selectively.


## Prompt Engineering Observation
---

The prompt was modified in the final hybrid experiment to make the VLM choose more clearly from the candidate list.

The final run had:

- 0 invalid outputs,
- strong candidate-list following,
- 66 fixed classifier errors,
- and a +3.8 percentage-point hybrid accuracy gain.


## Key Findings
---

1. **ConvNeXt is the best standalone model for fixed-label classification.**  
   It is faster, more accurate, and produces useful probability outputs.

2. **The VLM is surprisingly strong zero-shot.**  
   It reached 80.6% accuracy without fine-tuning, but it was much slower.

3. **Forced-choice VLM scoring is educational but expensive.**  
   It gives classifier-like scores but requires many image-text scoring passes.

4. **The VLM is useful as a second-stage reviewer.**  
   When ConvNeXt is uncertain, the VLM can rerank candidate labels and fix many mistakes.

5. **Routing matters.**  
   The VLM can also regress correct predictions, so it should not blindly override ConvNeXt everywhere.

6. **The realistic hybrid system improved accuracy.**  
   Routing 17.4% of samples to the VLM improved accuracy from 87.5% to 91.3%.

---

## Final Conclusion

For this Food101 20-class classification task, the best system is not a standalone VLM. The best system is a hybrid:

```text
ConvNeXt as the fast primary classifier
+
VLM as a selective reranker for uncertain cases
```

The fine-tuned ConvNeXt handles most images efficiently. The VLM is only used when ConvNeXt is uncertain, using the classifier's top candidate labels as the candidate set.

This gives a practical balance:

- better accuracy than ConvNeXt alone,
- much less VLM cost than running the VLM on every image,
- and a more realistic use of VLMs for classification workflows.

The final hybrid experiment improved accuracy from 87.5% to 91.3% while routing only 17.4% of the validation set to the VLM.
