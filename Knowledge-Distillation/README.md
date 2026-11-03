# Knowledge distillation for ViT-based cyclist detection on edge hardware

Can a Vision Transformer small enough to run on a car's blind-spot camera still catch cyclists? This project distils a 5.52M-parameter ViT-Tiny from ViT-Large and ViT-Base teachers and compares three distillation signals under conditions that make the problem hard on purpose: cyclists are about 1% of the data, and the inputs are 32x32.

The short answer is that the simplest method wins. Logit distillation reached 0.922 F1 on the rare class against 0.856 for the same student trained without a teacher, at under 2% of the teacher's FLOPs. Matching the teacher's intermediate features or its attention maps did not beat it, and neither did routing the transfer through a mid-sized teacher assistant.

This was my final project for Efficient AI (Fall 2025).

## Results

Bicycle-class metrics on the 10,000-image CIFAR-100 test set. Every student is the same ViT-Tiny, so the cost columns do not move: distillation changes the weights, not the architecture.

| Experiment | Model | Precision | Recall | F1 | Params (M) | GFLOPs |
| --- | --- | --- | --- | --- | --- | --- |
| - | Teacher, ViT-Large | 1.000 | 0.960 | 0.980 | 303.3 | 59.7 |
| - | Teacher, ViT-Base | 0.931 | 0.950 | 0.941 | 85.8 | 16.9 |
| - | Student, no distillation | 0.920 | 0.800 | 0.856 | 5.52 | 1.08 |
| I | Logit KD, from ViT-Large | 0.966 | 0.860 | 0.910 | 5.52 | 1.08 |
| I | Feature KD, from ViT-Large | 0.956 | 0.860 | 0.905 | 5.52 | 1.08 |
| I | Attention KD, from ViT-Large | 0.945 | 0.860 | 0.901 | 5.52 | 1.08 |
| II | Logit KD, from ViT-Base | 0.967 | 0.880 | **0.922** | 5.52 | 1.08 |
| II | Feature KD, from ViT-Base | 0.917 | 0.880 | 0.898 | 5.52 | 1.08 |
| II | Attention KD, from ViT-Base | 0.966 | 0.840 | 0.898 | 5.52 | 1.08 |
| III | Logit KD via ViT-Small assistant | 0.944 | 0.850 | 0.895 | 5.52 | 1.08 |

Accuracy is not reported as evidence anywhere: at 1% positives, answering "no bike" every time scores 99%.

Three things worth pulling out of that table:

- **Distillation buys precision more than recall.** Recall moves from 0.800 to 0.860 and then stops; precision moves from 0.920 to as high as 0.967. For a blind-spot warning, fewer false alarms is the difference between a system a driver trusts and one they switch off.
- **The weaker teacher produced the better student.** ViT-Base has 28% of ViT-Large's parameters and is 0.039 F1 worse itself, yet its logit-KD student is the best in the project. The teacher's job here seems to be supplying a clean decision boundary, and ViT-Base's is already clean enough.
- **Narrowing the capacity gap did not pay.** Not by shrinking the teacher, not by adding an assistant. The assistant chain cost twice the training time and landed 0.015 below direct transfer from the same teacher.

## What's in here

```
notebooks/vit_kd_bicycle_detection.ipynb   all three experiments, with the outputs from my run
report/report.pdf                          the write-up: method, results, discussion, limitations
report/report.md                           the same write-up as markdown, so it reads on GitHub
slides/efficient_ai_presentation.pptx      the presentation deck
results/results.csv                        every test-set number in the tables above
requirements.txt
```

The notebook is the whole pipeline, in the order the report describes it: data, then Experiment I (ViT-Large teacher, three signals), Experiment II (ViT-Base teacher, same three), Experiment III (teacher assistant). It was originally two notebooks, which is why the tidier merged version is what is here.

The write-up is a revised version of what I submitted for the course: the cost columns in the tables are now complete, and the discussion of run-to-run variance and of the pipeline defect noted below was added afterwards.

## Running it

It needs a GPU. I used Colab with an A100; Parts 1 and 3 take about four hours together, Part 2 about ninety minutes, almost all of it in the ViT-Large forward passes.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/vit_kd_bicycle_detection.ipynb
```

CIFAR-100 downloads itself on the first run. Cells are meant to run top to bottom; each training cell saves its best checkpoint and the cell after it evaluates that checkpoint on the test set. Every cell still carries the output from my run, so the numbers can be read without running anything.

## Method, briefly

CIFAR-100 is relabelled binary: class 8 (bicycle) is positive, the other 99 classes are negative, which leaves 443 positives in the training split and 100 in the test set. Images are upsampled to 224x224 for the pretrained ViT backbones. Students train on a loader with positives oversampled 8x and a class weight of 4.0; teachers train on the natural 1% distribution with a class weight of 20.0, so their soft labels are not shaped by my resampling. Model selection is always on validation F1 for the bicycle class.

The three signals:

- **Logit KD.** KL against the teacher's temperature-softened outputs (T = 4.0), mixed with weighted cross-entropy at alpha = 0.7.
- **Feature KD.** L2 between the CLS tokens of both models at block 3 and the last block, with the teacher's token projected down to the student's width of 192 by a learnable layer trained alongside the student.
- **Attention KD.** L2 between the normalised CLS-to-patch attention rows of the last block. `timm` fuses attention, so the map cannot be read off a forward hook; the hook recomputes `q @ k^T` from the block's input tokens. Head counts differ (16 against 3), so the maps are averaged over heads before being compared.

Everything else is held fixed across runs: 18 epochs for students, AdamW at 2e-4 with weight decay 0.05, a cosine schedule stepped per batch, batch size 32, gradient clipping at norm 1.0.

## Known issues

Two things I would want a reader to know rather than discover.

**The validation split was augmented.** The train and validation subsets wrap the same CIFAR-100 object, so setting a transform on one sets it on both, and the last assignment wins. Validation images were therefore evaluated with the training augmentation applied. This affected which checkpoint got picked at each epoch, and it makes the validation F1 numbers in the logs pessimistic. The test set is a separate object and was always evaluated clean, so the reported results stand. I found this while cleaning the code up for release and documented it rather than quietly re-running, because the fix changes which checkpoint is selected and therefore every number above.

**Single runs, not averages.** Experiment II retrains the baseline student, and the two baseline runs disagree under the same seed: 0.883 against 0.856. That 0.027 spread comes from dataloader ordering and cuDNN kernel choice, and it is about the same size as the differences between the three distillation methods. The logit-KD advantage holds under both teachers and is big enough to survive it; the exact ordering of feature against attention KD is not something I would claim. Averaging over seeds is the first thing I would add with more compute.

## License

MIT, see `LICENSE`.
