# Efficient Knowledge Distillation for Vision-Based Blind-Spot Detection in Resource-Constrained Environments

Mehrdad Mirpourian

## Abstract

Cyclists are hard to see from a car, and hardest of all in the blind spot, where mirrors do not reach. Vision models can close that gap, but the models that work best are far too large for the hardware a car actually carries. This project asks how much of a large Vision Transformer's ability to spot a cyclist can be pushed into a model small enough to deploy, and which kind of distillation signal does the pushing best.

I reformulate CIFAR-100 into a binary bicycle-detection task in which positives make up roughly 1% of the data, and distil a ViT-Tiny student from ViT-Large and ViT-Base teachers using three signals: softened logits, intermediate CLS-token features, and CLS-to-patch attention maps. I also test a teacher-assistant chain with ViT-Small in the middle.

Every distillation method improves the student over an undistilled baseline, with the gain concentrated in precision. The simplest signal does best: logit distillation reaches an F1 of 0.910 from the ViT-Large teacher and 0.922 from the ViT-Base teacher, against 0.856 for the baseline student. Neither matching the teacher's internal features nor routing the transfer through an assistant beats it. For a 5.52M-parameter student at 32x32 input resolution, a sharper decision boundary appears to be more useful than a richer representation.

## 1. Introduction

Cycling has grown quickly in dense cities, and the injury numbers have grown with it. In 2024 roughly 1,500 cyclists in New York City were injured in crashes involving driver distraction, with hundreds more from failure to yield and from dooring [Goldwater, 2025]. A disproportionate share of these happen in the vehicle's blind spot, the region a driver structurally cannot see. Blind-Spot Detection and Warning Systems (BSDWS) exist to cover it.

Most production systems combine sensors. Automotive radar measures range and relative speed reliably, and keeps working in rain and darkness [Liu et al., 2017], but it is weak at saying *what* it has found: a cyclist and a roadside post can look much the same in radar space. Camera-based models supply that missing semantic layer [Aquilina and Gatt, 2023; Huu, 2023]. The difficulty is that the vision models with the best accuracy, Vision Transformers among them, assume compute budgets that automotive edge platforms do not have. A module running on an embedded processor shares its power and thermal envelope with everything else in the car and has a latency budget measured in milliseconds.

Knowledge distillation (KD) is the standard response to that squeeze: train a small student to reproduce the behaviour of a large teacher, and deploy only the student [Hinton et al., 2015; Cho and Hariharan, 2019; Ji et al., 2021]. Which *part* of the teacher's behaviour the student should reproduce is less settled, and the answer plausibly depends on the task. Blind-spot detection has two properties that make it an unusual case: the object of interest is rare, and the input is often small and poor, a few dozen pixels of cyclist at the edge of a wide-angle frame.

This project isolates those two properties. CIFAR-100 becomes a binary task, bicycle against all 99 other classes, which leaves about 1% positives at 32x32 resolution. I then run the same student architecture through five training regimes under one fixed budget and compare them on the metric that matters for a safety system, F1 on the rare class.

The rest of this write-up is organised as follows. Section 2 reviews radar and vision approaches to blind-spot sensing and the distillation literature. Section 3 describes the method, Section 4 the three experiments and the training setup. Section 5 reports results, Section 6 discusses them, Section 7 sets out what this design cannot tell us, and Section 8 concludes.

## 2. Related work

**Radar-based detection.** Early blind-spot systems were radar systems. Radar measures range directly, relative velocity through Doppler shift, and bearing, and is indifferent to lighting. Chen and Chen [2009] showed real-time detection of approaching vehicles was feasible, which opened a long line of work on picking moving targets out of cluttered returns: multistatic detection under unknown coloured Gaussian interference [Shtarkalev and Mulgrew, 2015], dynamic subspace methods [Du and Zhang, 2014], improved Gaussian mixture models [Chen et al., 2015], and multidimensional digital smoothing filters [Kennedy, 2015]. Reported performance under controlled conditions is strong [Liu et al., 2017]. The persistent weakness is semantic: classes with near-identical radar signatures are hard to separate, and a cyclist is one of them.

**Vision-based detection.** Convolutional and transformer models supply the discrimination radar lacks, and have been applied to vulnerable road users including cyclists and motorcyclists [Aquilina and Gatt, 2023; Huu, 2023; Dosovitskiy et al., 2020]. They support classification and scene understanding at a level radar cannot reach, which is why hybrid camera-radar designs are common [Nezhad et al., 2021]. Their costs are sensitivity to weather and lighting, and a compute load that embedded automotive platforms cannot absorb.

**Knowledge distillation.** KD trains a compact student against the soft outputs of a larger teacher, which carry more information than hard labels alone [Hinton et al., 2015]. Later work moved beyond output matching to intermediate features and attention maps, and added teacher-assistant schemes that insert a middle model to reduce the capacity gap between teacher and student [Cho and Hariharan, 2019; Mirzadeh et al., 2020; Ji et al., 2021]. How well any of these transfers is known to depend on the task and the data distribution, which is what motivates a controlled comparison in the blind-spot setting rather than relying on results from balanced, high-resolution benchmarks.

## 3. Method

### 3.1 Task and dataset

CIFAR-100 contains 60,000 colour images at 32x32, 50,000 for training and 10,000 for test, over 100 classes. I keep class 8, bicycle, as the positive class and merge the remaining 99 into a single negative class. Positives then make up about 1% of each split: 500 of the 50,000 training images and 100 of the 10,000 test images.

CIFAR-100 images are obviously not automotive camera frames. What this reformulation buys is a tractable environment with the two properties I want to study held at realistic severity: a safety-critical class that is rare, and inputs too small to carry fine detail. It lets me compare distillation signals under those conditions on a single GPU, which a high-resolution driving dataset would not have allowed within the project's compute budget. Section 7 returns to what this costs in external validity.

Of the 50,000 training images, 5,000 are held out for validation and model selection, which leaves 443 positives for training and 57 for validation. The 10,000-image test set, with its 100 positives, is used once per model for the numbers reported in Section 5.

### 3.2 Class imbalance

At 1% positives, a model that answers "no bike" unconditionally is 99% accurate. Accuracy is therefore not reported anywhere as evidence; all model selection and all reported results use precision, recall and F1 on the bicycle class.

Imbalance is handled in two ways. Student runs draw from a loader in which positive indices are repeated eight times, which raises the positive rate from about 1% to about 7.4% without collapsing to an even split, and use a cross-entropy class weight of 4.0 on positives. Teachers train on the natural, unresampled distribution with a heavier class weight of 20.0. Keeping the teachers on the real distribution means their soft labels are not shaped by my resampling choice, so what the students learn from is the teacher's view of the actual class prior.

### 3.3 Augmentation

Training images are resized to 224x224, then augmented with horizontal flips (p = 0.5), random cropping with four-pixel padding, rotation up to five degrees, colour jitter on brightness, contrast and saturation, random greyscale (p = 0.05), additive Gaussian noise at 0.005, and random erasing (p = 0.15). The geometric and photometric parts stand in for viewpoint and lighting variation; the noise and erasing stand in for sensor noise and partial occlusion, both of which a real blind-spot camera produces constantly.

The noise is deliberately mild. In distillation the teacher and the student see the same augmented image, and heavy augmentation makes the teacher's soft label a poorer description of what the student is looking at. Validation and test images are only resized and normalised.

### 3.4 Architectures

All models are ImageNet-pretrained ViTs from `timm` with a patch size of 16 at 224x224 input, so teacher and student share an architecture family and differ only in capacity. The classification head of each is replaced with a two-class linear layer.

| Role | Model | Parameters | GFLOPs | Width | Heads |
| --- | --- | --- | --- | --- | --- |
| Teacher, Experiments I and III | ViT-Large | 303.3M | 59.7 | 1024 | 16 |
| Teacher, Experiment II | ViT-Base | 85.8M | 16.9 | 768 | 12 |
| Assistant, Experiment III | ViT-Small | - | - | 384 | 6 |
| Student, all experiments | ViT-Tiny | 5.52M | 1.08 | 192 | 3 |

Parameter counts and FLOPs are measured with `fvcore` for a single 224x224 image. `fvcore` does not recognise the fused attention and GELU kernels, so the GFLOP figures are lower bounds; they are used only to compare the models against each other. The assistant is not costed because it is never deployed: it exists to train the student and is discarded. The deployed model in every configuration is the same ViT-Tiny, 55x smaller than the ViT-Large teacher and about 1.8% of its FLOPs.

### 3.5 Distillation strategies

**Logit distillation.** The student matches the teacher's temperature-softened output distribution through a KL term, alongside weighted cross-entropy on the true labels:

```
L = alpha * T^2 * KL(student/T || teacher/T) + (1 - alpha) * CE(student, y)
```

with T = 4.0 and alpha = 0.7. The T^2 factor restores the gradient scale that temperature softening removes.

**Feature distillation.** The student matches the teacher's internal representation rather than its conclusion. I read the CLS token at block 3 and at the final block of both models through forward hooks, project the teacher's token down to the student's width of 192 with a learnable linear layer, and penalise the L2 distance. Logit KD and cross-entropy remain in the objective:

```
L = 0.5 * L_logit_KD + 1.0 * L_feature + 0.5 * CE
```

Two layers rather than one gives the student both an early and a late view of the teacher's representation. The projection trains jointly with the student and is discarded afterwards.

**Attention distillation.** The student is pushed to attend where the teacher attends. I take the CLS-to-patch attention row from the last block of each model, renormalise both, and penalise the squared difference:

```
L = 0.5 * L_logit_KD + 1.0 * L_attention + 0.5 * CE
```

Two implementation details matter here. `timm` uses fused attention, so the attention matrix is never exposed as a module output and cannot be lifted from a hook; the hook recomputes `q @ k^T` from the block's input tokens instead. And the teacher has 16 heads to the student's 3, so there is no head-to-head correspondence. Averaging over heads before comparing gives both models maps of the same shape, which is the standard way around this mismatch in the ViT distillation literature. Renormalising matters too: without it, part of the loss measures a difference in total attention mass rather than a difference in where the attention goes.

**Teacher-assistant distillation.** Following Mirzadeh et al. [2020], knowledge moves in two hops. ViT-Large first teaches a ViT-Small assistant; the assistant, frozen, then teaches ViT-Tiny. The student never sees the ViT-Large teacher. Both hops use logit KD with weights 0.7 on the KD term and 0.3 on cross-entropy.

## 4. Experimental setup

Three experiments, each a controlled comparison in which one thing changes:

**Experiment I: distillation across a large capacity gap.** ViT-Large teaches ViT-Tiny, with the three signals compared head to head against an undistilled baseline. This establishes which signal survives a 55x capacity gap.

**Experiment II: teacher capacity.** The same three runs with ViT-Base as teacher. The question is whether a less capable but architecturally closer teacher produces a better student, and in particular whether feature matching gets easier when the projection bridges 768 to 192 rather than 1024 to 192.

**Experiment III: multi-stage transfer.** The teacher-assistant chain, testing whether a staged hand-off through ViT-Small beats direct transfer from ViT-Large.

Shared training setup, held identical across every run so that the comparisons are about the distillation signal and nothing else:

| Setting | Teachers | Students and assistant |
| --- | --- | --- |
| Epochs | 4 | 18 |
| Optimiser | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 |
| Learning rate | 2e-5 | 2e-4 |
| Schedule | cosine to 1e-6, stepped per batch | cosine to 1e-6, stepped per batch |
| Batch size | 32 | 32 |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Positive class weight | 20.0 | 4.0 |
| Sampling | natural (~1% positive) | positives oversampled 8x |
| Selection | best validation bike-class F1 | best validation bike-class F1 |

The cosine schedule is stepped once per batch rather than once per epoch, as the ViT fine-tuning recipes assume. Seeds are fixed at 42 for Python, NumPy and Torch, and 41 for the train/validation split. Runs were on a single Colab A100.

## 5. Results

All figures are bicycle-class metrics on the 10,000-image test set, from the checkpoint with the best validation F1. Students are ViT-Tiny in every row, so the parameter and FLOP columns are constant: distillation changes the weights, not the architecture, which is the point of the method.

### 5.1 Experiment I: ViT-Large teacher

| Model | Precision | Recall | F1 | Params (M) | GFLOPs |
| --- | --- | --- | --- | --- | --- |
| Teacher (ViT-Large) | 1.000 | 0.960 | 0.980 | 303.3 | 59.7 |
| Student (ViT-Tiny), no KD | 0.920 | 0.800 | 0.856 | 5.52 | 1.08 |
| KD (logit) | 0.966 | 0.860 | **0.910** | 5.52 | 1.08 |
| KD (feature) | 0.956 | 0.860 | 0.905 | 5.52 | 1.08 |
| KD (attention) | 0.945 | 0.860 | 0.901 | 5.52 | 1.08 |

All three signals beat the baseline, and the ordering is logit, then feature, then attention, though the three distilled students sit within 0.009 F1 of one another. The structure of the gain is more informative than the ordering: recall rises from 0.800 to 0.860 and then stops there for all three methods, while precision climbs from 0.920 to between 0.945 and 0.966. Distillation is mostly teaching the student to stop calling non-cyclists cyclists. The student closes a little under half the F1 gap to its teacher, at 1.8% of the teacher's FLOPs.

### 5.2 Experiment II: ViT-Base teacher

| Model | Precision | Recall | F1 | Params (M) | GFLOPs |
| --- | --- | --- | --- | --- | --- |
| Teacher (ViT-Base) | 0.931 | 0.950 | 0.941 | 85.8 | 16.9 |
| Student (ViT-Tiny), no KD, this session | 0.943 | 0.830 | 0.883 | 5.52 | 1.08 |
| Student (ViT-Tiny), no KD, Experiment I | 0.920 | 0.800 | 0.856 | 5.52 | 1.08 |
| KD (logit) | 0.967 | 0.880 | **0.922** | 5.52 | 1.08 |
| KD (feature) | 0.917 | 0.880 | 0.898 | 5.52 | 1.08 |
| KD (attention) | 0.966 | 0.840 | 0.898 | 5.52 | 1.08 |

A teacher with 28% of the parameters and a 0.980-to-0.941 deficit in its own F1 did not produce a worse student. Logit KD from ViT-Base reaches 0.922, the best student in the project, and the ranking of the three signals is unchanged from Experiment I, so that ranking is not an artefact of one teacher.

The gap between logit KD and the other two is wider here, and feature matching did not improve despite the shorter projection. That is some evidence that the binding constraint is the student's own width rather than the size of the dimensional jump.

One caveat belongs with this table rather than in a footnote. The baseline is listed twice because I retrained it in this session, and the two runs do not agree: 0.883 here against 0.856 in Experiment I, under the same seed and the same configuration. The difference comes from nondeterminism I did not control, in dataloader worker ordering and cuDNN kernel selection. Against the session's own baseline, logit KD from ViT-Base gains 0.039 F1 rather than 0.066. The run-to-run spread of 0.027 on identical settings is of the same order as the differences *between* the three distillation signals, which is why Section 6 treats the logit result as the safe conclusion and the exact ordering of feature and attention KD as unresolved.

### 5.3 Experiment III: teacher assistant

| Model | Precision | Recall | F1 | Params (M) | GFLOPs |
| --- | --- | --- | --- | --- | --- |
| Teacher (ViT-Large) | 1.000 | 0.960 | 0.980 | 303.3 | 59.7 |
| Student (ViT-Tiny), no KD | 0.920 | 0.800 | 0.856 | 5.52 | 1.08 |
| KD (logit), direct from ViT-Large | 0.966 | 0.860 | **0.910** | 5.52 | 1.08 |
| KD (logit + TA via ViT-Small) | 0.944 | 0.850 | 0.895 | 5.52 | 1.08 |

The assistant chain beats the undistilled baseline by a clear margin, 0.895 against 0.856, so staged transfer does work. It does not beat direct distillation from the same teacher. Given the variance discussed above, the 0.015 shortfall is not a result I would lean on hard; what the experiment does establish is that the assistant bought nothing here, at double the training cost, since the assistant has to be trained before the student can start.

## 6. Discussion

**A richer signal is not automatically a better one.** The clearest pattern across all three experiments is that logit distillation, the oldest and simplest method tested, produced the best student every time. The usual argument for feature and attention matching is that output probabilities are an impoverished target, especially so here, where two classes leave very little structure in the teacher's distribution for the student to learn from. The results point the other way, and the student's capacity is the likely reason. Asking a 192-wide model with 3 attention heads to reproduce a 1024-wide representation or a 16-head attention pattern sets a target it cannot represent, so the extra loss terms act partly as noise on an optimisation that is already hard at 1% positives. A softened decision boundary is something a model this small can actually absorb.

**Teacher capacity set the ceiling less than I expected.** Replacing ViT-Large with ViT-Base left both the absolute student quality and the ranking of the methods intact. For this task the teacher's job is apparently to supply a clean decision boundary, and ViT-Base at 0.941 F1 is already clean enough; the extra 0.039 of teacher F1 that ViT-Large brings does not reach the student. There is a practical corollary: the teacher does not have to be the largest model available, which matters when teacher fine-tuning is itself the expensive part of the pipeline.

**Closing the capacity gap did not help, by either route.** Neither shrinking the teacher nor inserting a ViT-Small assistant beat direct logit KD from ViT-Large. The teacher-assistant literature motivates the middle model by the capacity gap, but a gap in width is not the only thing in play when the target class is 1% of the data. Each hop in the chain is a lossy re-encoding of the decision boundary, and the assistant's own errors on the rare class are inherited by the student with no way to check them against the stronger teacher. For rare-class problems, direct access to the best available soft labels looks like the better trade.

**What I would deploy.** On this evidence, logit KD. It gave the best student under both teachers, needs no projection layers, no hooks and no recomputed attention, and is correspondingly easier to get right. For a blind-spot module the precision gain is the valuable part: a system that cries cyclist at empty road gets switched off or ignored, and every method here improved precision more than recall. The recall ceiling at 0.86 to 0.88 is the real remaining problem, and it is not one distillation solved: roughly one cyclist in eight is still missed, which no safety engineer would accept from a sole sensor. That is an argument for the sensor fusion the production systems already use, not against the compressed model.

## 7. Limitations and future work

**The dataset is a proxy, not a driving scene.** CIFAR-100 at 32x32 reproduces the class imbalance and the low effective resolution of the real problem, but none of its geometry: no perspective, no motion blur, no weather, no partial occlusion by the car's own pillar, and one object per image. The ranking of distillation signals found here should be re-tested on a driving dataset such as nuScenes, BDD100K or Cityscapes before it is used to choose a method for a real module. My specific expectation is that feature and attention matching have more to offer at higher resolution, where the teacher's spatial representation carries information that 32x32 inputs simply do not contain, and that is a testable prediction rather than a hedge.

**Single runs, not means.** Every number in Section 5 comes from one training run. The duplicate baseline in Experiment II shows the run-to-run spread on identical settings is about 0.027 F1, comparable to the differences between the three distillation methods. The logit-KD advantage is large enough and consistent enough across both teachers to survive that, but the exact ordering of feature against attention KD is not established, and I would not claim it is. Repeating each configuration across several seeds and reporting means with intervals is the first thing I would add.

**Hyperparameters were set by hand.** Temperature, the loss weights, the 8x oversampling factor, the class weights and the learning rates were chosen once and held fixed so the comparison stayed fair, not tuned per method. The fixed budget was a deliberate choice, but it means each method is being judged at settings that may suit it unequally: the feature and attention objectives each carry two extra weights that were never searched, and they may well be the methods most penalised by this.

**A known defect in the data pipeline.** The train and validation subsets wrap the same underlying CIFAR-100 object, so assigning a transform to one assigns it to both. The last assignment wins, which means the validation split was evaluated with the training augmentation applied. This affected model selection: the checkpoint chosen at each epoch was the one best on augmented validation images. The test set is a separate object and was always evaluated clean, so the reported results are sound, but the selection was noisier than intended and the validation F1 values in the logs understate clean performance. I found this while preparing the code for release and chose to document it rather than silently re-run, since the fix changes which checkpoint is picked and therefore every number in Section 5. Rerunning with the pipeline corrected is the other change I would make first.

**Static images, not video.** A real blind-spot system sees a stream and needs to reason about trajectory and closing speed, not just presence. Extending this to spatiotemporal models, and fusing the vision output with radar range and Doppler measurements rather than treating vision alone, is the direction that would make the work operationally relevant.

## 8. Conclusion

I set out to find how much of a large Vision Transformer's cyclist-detection ability can be moved into a model small enough for automotive edge hardware, and which distillation signal moves the most. Across three experiments on a 1%-positive binary reformulation of CIFAR-100, a 5.52M-parameter ViT-Tiny distilled with plain logit KD reached 0.910 to 0.922 F1 against 0.856 for the same architecture trained without a teacher, at under 2% of the teacher's FLOPs.

The result I did not expect is that the simplest method won consistently. Matching the teacher's intermediate features or its attention maps did not beat matching its softened outputs, under either teacher, and inserting a teacher assistant to narrow the capacity gap did not beat direct transfer. For a student this small, at this resolution, with a class this rare, the useful thing a teacher provides is a cleaner decision boundary rather than a richer internal representation. Practitioners building lightweight safety-critical vision modules should probably exhaust the simple method before paying the implementation cost of the complex ones.

## References

D. Aquilina and T. Gatt. Motorcycle blind spot detection through computer vision techniques. In *Proceedings of the International Symposium on Image and Signal Processing and Analysis (ISPA)*, pages 1-6, 2023.

C.-T. Chen and Y.-S. Chen. Real-time approaching vehicle detection in blind-spot area. In *Proceedings of the IEEE Intelligent Transportation Systems Conference*, pages 1-6, 2009.

X. R. Chen, C. L. Xi, and J. H. Cao. Research on moving object detection based on improved mixture Gaussian model. *Optik*, 126(20):2869-2874, 2015.

J. H. Cho and B. Hariharan. On the efficacy of knowledge distillation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

B. Du and L. Zhang. Target detection based on a dynamic subspace. *Pattern Recognition*, 47:344-358, 2014.

R. Goldwater. Bicycle accident statistics in NYC, 2025. Bicycle Accident Lawyers Group.

G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network, 2015.

P. N. Huu. Vehicle blind spot detection and tracking system based on machine learning model. In *Proceedings of the International Scientific Conference on Applying New Technology in Green Buildings (ATiGB)*, pages 292-297, 2023.

M. Ji, B. Heo, and S. Park. Show, attend and distill: Knowledge distillation via attention-based feature matching. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 35, pages 7945-7952, 2021.

H. L. Kennedy. Multidimensional digital smoothing filters for target detection. *Signal Processing*, 114:251-264, 2015.

G. Liu, L. Wang, and S. Zou. A radar-based blind spot detection and warning system for driver assistance. In *Proceedings of the IEEE Advanced Information Technology, Electronic and Automation Control Conference (IAEAC)*, pages 2204-2208, 2017.

S. I. Mirzadeh, M. Farajtabar, A. Li, N. Levine, A. Matsukawa, and H. Ghasemzadeh. Improved knowledge distillation via teacher assistant. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, 2020.

A. P. Nezhad, M. Ghatee, and H. Sajedi. Blind spot warning system based on vehicle analysis in stream images by a real-time self-supervised deep learning model, 2021. Authorea Preprints.

B. Shtarkalev and B. Mulgrew. Multistatic moving target detection in unknown colored Gaussian interference. *Signal Processing*, 115:130-143, 2015.
