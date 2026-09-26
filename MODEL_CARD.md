---
license: apache-2.0
license_source: "the Hugging Face card for the converted checkpoint; the upstream microsoft/CvT repository, which released the original weights, is MIT-licensed"
model_card_spec: "1.2"
pipeline_tag: image-classification
base_model: microsoft/cvt-13
date_published: "2021-03"
date_published_source: "month of the CvT paper (arXiv:2103.15808, submitted 2021-03-29); the dates the original weights and the Hugging Face conversion were first published are not established by this repository"
---

# CvT-13 (ImageNet-1k, 224 px) — Image Classification with Bounded Fine-Tuning

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Fcvt--13-ffcc4d?style=flat)](https://huggingface.co/microsoft/cvt-13)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2FCvT-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/CvT)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2103.15808-b31b1b.svg)](https://arxiv.org/abs/2103.15808)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream snapshot is pinned to Hub commit `84e365a5f6a5ca987486abb25f3d8e5265cdc44d`, and the manifest records every file's SHA-256. Default-path execution recorded on 2026-09-26 (Kaggle T4); REL12 BYOD exercise pending before promotion. The measured values under Metrics come from that one run: one seeded split of 60 held-out and 60 unseen CIFAR-10 thumbnails (frog and truck), one runtime, with near-duplicate leakage between the training and evaluation splits (see Metrics). They are tutorial evidence, not a benchmark.

---

## Interactive Colab Tutorials

- **End-to-end classification and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/cvt-classification-pipeline/blob/main/tutorials/cvt_classification_colab.ipynb) [`cvt_classification_colab.ipynb`](https://github.com/kurtvalcorza/cvt-classification-pipeline/blob/main/tutorials/cvt_classification_colab.ipynb)
  *A pinned CIFAR-10 frog/truck subset, ImageNet top-5 predictions, majority-class and zero-shot ImageNet baselines, degenerate-input probes, a bounded fine-tune of a new two-class head, held-out and unseen-split scores, and SafeTensors adapter export and reload.*

---

#### Description

`microsoft/cvt-13` is CvT-13 from "CvT: Introducing Convolutions to Vision Transformers" (Wu et al., arXiv:2103.15808), converted to Hugging Face Transformers. The upstream card states that it was pre-trained on ImageNet-1k at 224×224. The upstream `microsoft/CvT` repository reports 20M parameters, 4.5 GFLOPs and 81.6 ImageNet top-1 accuracy for this model. A test in this repository counts 19,997,480 parameters in the architecture it builds.

CvT is a Vision Transformer with two convolutional parts. Each of its three stages starts with a convolutional token embedding: an overlapping strided convolution (7×7 with stride 4, then 3×3 with stride 2 twice) that shrinks the token map and widens the channels to 64, 192 and 384. Each attention layer computes its queries, keys and values through a convolutional projection, a depthwise 3×3 convolution followed by BatchNorm, with keys and values subsampled by stride 2. The convolutions carry spatial position, so there are no position embeddings. CvT-13 has 1, 2 and 10 blocks per stage, and a `[CLS]` token enters in the last stage. LayerNorm on that token and a linear head output a softmax over the 1000 ImageNet-1k classes. The snapshot's `config.json` declares `CvtForImageClassification`, which is how the pipeline loads it.

The preprocessing is the snapshot's `ConvNextFeatureExtractor` setting: size 224 with `crop_pct` 0.875, so a bicubic resize of the shorter side to 256 px and a 224 px centre crop, then the ImageNet mean and standard deviation. The pipeline re-implements it with torchvision, and a test checks the result against the Transformers `ConvNextImageProcessor` loaded from the committed `preprocessor_config.json`.

This repository adds gradient fine-tuning on a caller's labelled images. `from_pretrained(class_names=...)` replaces the 384-to-1000 `classifier` layer with a new layer for the caller's classes. `finetune` then trains with cross-entropy, either the whole network (the default) or the new head alone with the backbone frozen and its BatchNorm statistics held.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the pinned files, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `CvtPipeline.from_pretrained`: construction of `CvtForImageClassification` from the verified `config.json`, then `load_state_dict(strict=True)` from the verified SafeTensors file, and a check that the config's labels at the zero-shot group indices are the expected ImageNet classes;
- `predict`: input checks and top-k softmax scores; `zero_shot_evaluate`: a baseline that maps groups of ImageNet classes onto task labels without training;
- `fetch_sample_archive`, `read_class_archive`, `read_class_folder`, `validate_dataset`, `split_dataset`, `validate_inputs` and `evaluation_report`: the data acquisition, validation and single-batch evaluation stages;
- `finetune`, `evaluate` (accuracy, balanced accuracy, per-class recall, confusion matrix, majority-class baseline), `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is closed-set, single-label image classification. `predict` takes one image or a batch of up to 64 and returns, per image, the top-k classes with their softmax scores and the argmax label.

The pretrained head fits photographs of the objects, animals and scenes ImageNet-1k names. The adaptation path fits a small labelled set of a few classes, for example product categories, plant or specimen types, or document page types, where a new head on a pretrained backbone is enough.

The intended role is a reference convolution-augmented Vision Transformer and a teaching baseline for transfer learning. CvT-13 is small enough to fine-tune on a free GPU in minutes. A reader's own application can embed `CvtPipeline` as a classifier or, with the head removed, as a `[CLS]`-token image feature extractor.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- the pretrained vocabulary is the 1000 ImageNet-1k classes; anything else receives the nearest of them;
- a `score` is a softmax value under the model's training distribution, not a calibrated probability on the user's images;
- the classifier has no reject option: every image, including a blank one, receives a class;
- accuracy is only meaningful next to the majority-class baseline of the same data, and balanced accuracy is the fairer summary when classes are imbalanced;
- sketches, screenshots, medical, aerial and thermal images, and very small images upsampled to 224 px, are distribution shifts from ImageNet photographs;
- a fine-tune on a few hundred images demonstrates the workflow and does not produce a deployable classifier;
- in the recorded tutorial run the adapted frog/truck head answered `frog` with score 0.546 for a blank image and 0.683 for pure noise, and the ImageNet head gave both CIFAR-10 frog sample images non-frog top-1 labels (`screw`, `platypus`);
- the tutorial's split is by image ID only and the sample archive pairs each `original_images` file with a pixel-identical `darkened_images` file, so in the recorded run 46 of 60 held-out and 36 of 60 unseen images had their counterpart in the training split; the tutorial's held-out and unseen scores are not evidence of generalisation.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside the loaded vocabulary, no text prompts, no multi-label output, no localisation (detection or segmentation), and no calibrated confidence or open-set rejection.
2. **Input boundary:** `predict` and `validate_inputs` reject anything that is not a `PIL.Image.Image` (`TypeError`), batches above `MAX_BATCH = 64`, and image sides below `MIN_IMAGE_SIDE = 8` px or above `MAX_IMAGE_SIDE = 4096` px (`ValueError`). `top_k` must be between 1 and the number of classes.
3. **Input boundary:** every image is resized so its shorter side is 256 px and centre-cropped to 224×224. Content outside the central crop is not seen, and small images are upsampled. The tutorial's 32×32 CIFAR-10 images are an example of the latter, not a recommended input size.
4. **Data boundary for adaptation:** `validate_dataset` accepts up to 5,000 records (`MAX_RECORDS`) over 2 to 1,000 classes, with at least 2 images per class. The bounded tutorial fine-tune (5 epochs, no augmentation) is not a training recipe for a production classifier.
5. **Decision boundary:** not for decisions that act on labels without a person reviewing them, including medical, safety, legal, hiring, credit or law-enforcement decisions. Any use also requires accuracy and per-class recall measured locally on the deployment's own labelled images.

#### Factors

###### Groups

ImageNet-1k names only three person classes (`ballplayer`, `groom` and `scuba diver`), but many of its images show people, and classes such as clothing, sports equipment and musical instruments correlate with them. The pretrained model can therefore produce labels whose errors differ with the skin tone, age, gender presentation or dress of the people in an image; this was not evaluated here or, as far as this repository knows, upstream.

ImageNet's photographs were collected from web search in a limited set of languages and regions. Objects, foods, animals and scenes from under-represented regions may be recognised less often; that is unmeasured.

An adapted model inherits whatever group structure the caller's labelled data has. The operator who classifies images of people, with the pretrained or an adapted model, owns a per-group audit on their own images before relying on the output.

###### Instrumentation

ImageNet-1k images are web photographs of varied resolution, mostly taken with consumer cameras. Inference images arrive from whatever produced them: phones, scanners, microscopes, drones or rendering engines.

Resolution, blur, compression, exposure, colour balance and framing all change the evidence. The resize-and-crop preprocessing discards the border of every non-square image and resamples every image to 224 px. The pipeline checks only type and size; it cannot detect a rotated scan, a thumbnail, a rendered image or an empty frame.

The tutorial sample is itself an instrument: CIFAR-10 images are 32×32 pixels, collected by Krizhevsky and labelled by hand, and are upsampled about sevenfold here. Labels a caller supplies carry whatever error their annotation process has, and a fine-tune learns a systematic labelling error as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. One run with the pinned weights is recorded: Kaggle Tesla T4, 2026-09-26 UTC, torch 2.14.0+cu130 (CUDA 13.0), torchvision 0.29.0+cu130, Transformers 4.57.6, `cuda:0`. The whole notebook took 279.5 s wall including installs, one kernel restart and the 80 MB snapshot download; the fine-tune cell (5 epochs on 280 images) took about 20 s. No memory or throughput figure was measured.

**Data environment.** The pretrained head assumes a photograph centred on one ImageNet object or scene. An adapted model assumes inference images that resemble its training images in source, framing and resolution. The tutorial's adaptation data is CIFAR-10 thumbnails, so a model adapted on it transfers to images of that kind and to little else. When these assumptions fail, the model still returns a class. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`evaluate(records, majority=...)` reports, for this pipeline's head on labelled records:

- `accuracy`: the fraction of images whose argmax label is correct;
- `balanced_accuracy`: the mean of the per-class recalls, which a model cannot raise by favouring the larger class;
- `per_class_recall` and `confusion_matrix` (rows true, columns predicted), which show where the errors go;
- `majority_baseline_accuracy`: the accuracy of always answering `majority`, which the caller sets to the training split's most frequent class.

`zero_shot_evaluate(records, groups)` scores the unmodified ImageNet head on task labels. It assigns each image to the label whose group of ImageNet classes holds the most softmax mass, and reports the same measures. For the tutorial, `frog` maps to ImageNet's three frog classes and `truck` to its big-truck classes; pickups are left out because CIFAR-10's `truck` excludes them.

`evaluation_report(result, truth, groups=...)` covers one batch of ImageNet-head predictions. It reports how often the true group appears in the top-1 and the top-k, with the verdict `sample-sanity`. Without labels it returns `not-measurable` and names the labelled data that would be needed.

The upstream repository reports ImageNet-1k top-1 accuracy 81.6 for CvT-13 at 224×224. Those values are upstream-reported, and this repository does not reproduce them.

Values measured by this repository (one run on Kaggle Tesla T4, 2026-09-26 UTC; exact notebook blob `5d9e491cd649`, commit `5e0f03b`; one pass, no dispersion estimate):

- **ImageNet head on 4 sample images** (CIFAR-10, 32 px upsampled): both trucks got `moving van` top-1 (0.738, 0.237); the two frogs got `screw` (0.288) and `platypus` (0.286). `sample-sanity` group hit rate 0.5 at top-1 and at top-5. This is a four-image check, not an ImageNet evaluation.
- **Held-out** (60 images, 30 per class, split seed 0 from 400 images), accuracy / balanced accuracy: majority class 0.500 / 0.500; zero-shot ImageNet mapping 0.9833 / 0.9833; untrained two-class head 0.250 / 0.250; fine-tuned 1.000 / 1.000. The zero-shot mapping already reaches 0.9833, so the fine-tune's gain is one image (about 1.7 points), which is within the noise of this split.
- **Unseen** (60 images): fine-tuned 0.9833 / 0.9833 (one frog predicted `truck`).
- **Fine-tune:** full model, 19,613,250 parameters trained, 5 epochs, batch 16, AdamW lr 0.0001; epoch losses 0.4279, 0.2589, 0.2276, 0.1724, 0.1632.
- **Degenerate probes:** the ImageNet head gave a blank image top-1 0.002 (`nematode`) and noise 0.016 (`jellyfish`); the adapted two-class head gave a blank image `frog` 0.546 and noise `frog` 0.683.
- **Adapter reload:** 60 images compared, tolerance 0.0001, equivalent.

**Near-duplicate leakage.** The sample archive holds 100 groups of pixel-identical images (an `original_images` file and its same-numbered `darkened_images` file), and the tutorial's split is by image ID only, not duplicate-aware. By the IDs in the run's predictions file, 46 of 60 held-out and 36 of 60 unseen images have their counterpart in the training split (a count of counterparts, not of confirmed pixel copies). The held-out and unseen scores are therefore not evidence of generalisation. The BYOD branches were not exercised in this run.

###### Decision thresholds

There is no score threshold. The reported label is the argmax over the head's classes, so every image receives exactly one of them. `top_k` changes how many ranked classes are returned, not which one is reported.

No acceptance threshold on accuracy is set anywhere in the repository. A deployment that needs to abstain must add its own rule, for example a minimum top-1 score chosen on labelled images from the deployment, and must measure what that rule costs in coverage. Re-check any such rule after a change of camera, image source, class set or adapter.

###### Approaches to uncertainty and variability

Every score, including the recorded ones, is one pass over one split: no repeated runs, no cross-validation, no bootstrap and no confidence interval. With the default 200 images per class, the tutorial's held-out split has 60 images, so one image moves accuracy by about 1.7 percentage points, and differences of a few points between methods are within noise.

Sources of run-to-run variability:

- the per-class sample drawn from the archive and the split, controlled by `DATASET_SEED` and `SEED`;
- the head initialisation and the batch order, controlled by the `seed` argument;
- stochastic depth in the last stage (the snapshot's config sets `drop_path_rate` to 0.1 there and 0.0 elsewhere) and the BatchNorm statistics of the convolutional projections, both active during full fine-tuning;
- GPU kernel selection, which is not forced to be deterministic, so repeated GPU runs can differ slightly.

A `score` is a softmax output, not a calibrated probability. A caller who needs calibrated confidence must fit a calibration map on labelled images from the deployment. A caller who needs an uncertainty estimate must evaluate on more images, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The upstream card states that the model was pre-trained on ImageNet-1k. ImageNet's images were gathered by web search and labelled by crowd workers; they include identifiable people and private settings, and published audits of the wider ImageNet collection have documented offensive and stereotyped labels in its person categories. Personal data is present in the training corpus by construction; it was not audited here.

The tutorial downloads `CIFAR-10-subset.zip` from the `Cleanlab/cifar-10-subset` dataset at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75` (MIT licence, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`). It keeps the `frog` and `truck` folders. CIFAR-10 is a set of 32×32 colour images labelled with ten object classes.

This repository distributes code, tests, documentation and one configuration file copied from the upstream snapshot (`preprocessor_config.json`, kept so a test can check the preprocessing against it). It does not distribute `model.safetensors` or the sample archive; both are downloaded at runtime and git-ignored.

An exported adapter contains weights fitted to the caller's training images. It does not contain the images, but it can reflect them. The operator must audit the images they classify, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, the upstream authors, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: triage of medical or dermatology photographs, sorting of images of people, content moderation, and quality inspection that stops a production line. Any of them would be admissible only with human review of every acted-on label. They would also need locally measured accuracy and per-class recall stratified by the groups named above, a documented abstention and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** if `MODEL_REVISION` were reset to `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` would raise before any download or model import. At the pinned revision, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants. It fetches only manifest-listed files, and only with `allow_download=True`. `verify_snapshot` checks every file's byte size and SHA-256 and refuses an entry with no recorded digest. `from_pretrained` builds the architecture from the verified `config.json` and loads the verified SafeTensors file with `strict=True`. The Hub repository also holds `pytorch_model.bin` (a pickle) and `tf_model.h5`; neither is staged or loaded, and the pin tool records their Hub LFS digests for provenance only.
- **Data integrity:** `fetch_sample_archive` downloads the sample at a fixed dataset commit and checks its size and SHA-256 before it is opened, with no fallback. `read_class_archive` refuses absolute member names and `..` segments and bounds the member count and the uncompressed size before decompressing anything; `read_class_folder` refuses files that link outside the directory.
- **Tests of those refusals:** tests assert that an unpinned package, a missing snapshot and a tampered digest are all refused before `torch`, `transformers` or `safetensors` is imported. Others assert that a full-size checkpoint strict-loads and that a missing tensor or a drifted config is refused, that the label check refuses a shifted ImageNet order, that the pipeline's transform equals the Transformers processor built from the committed `preprocessor_config.json`, and that a frozen fine-tune leaves every backbone weight and BatchNorm statistic unchanged.
- **Input integrity:** `validate_inputs` and `predict` share one checker for type, batch size, image size and `top_k`. `validate_dataset` rejects a record with missing keys, a non-image, an out-of-range image, an unknown label or a class with fewer than 2 images. It reports class imbalance and pixel-identical duplicates as findings.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-weights digest, class names and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key. It also refuses a different vocabulary, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for sampling, split, head initialisation and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no pickle deserialisation, no remote model code, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no class balancing, re-sampling or augmentation; `validate_dataset` reports imbalance and does not change it.

###### Risks and harms

- **A class for every image.** Blank, corrupted and out-of-domain images receive a label, often with a high score. Whoever acts on that label bears the harm; the tutorial probes a blank and a noise image and records what it finds.
- **Mislabelling within a closed vocabulary.** An object outside the vocabulary that resembles a class is labelled as that class, and nothing flags it.
- **Unequal error across groups.** Errors on images of people, and on objects from under-represented regions, may be more frequent; the people in the images bear that harm. It is unmeasured.
- **Overfitting in adaptation.** A fine-tune on a few hundred images can score well on a held-out split from the same source and fail on anything else. The operator who deploys it bears the harm whenever training and deployment images differ.
- **Misleading accuracy.** Accuracy on an imbalanced set can exceed the majority baseline by little while looking high. Reporting it without the baseline and balanced accuracy overstates quality.
- **Automation bias.** High softmax scores invite trust that an uncalibrated score has not earned. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of records that share a source photograph or session puts near-duplicates on both sides. The resulting held-out score overstates quality; `validate_dataset` reports exact duplicates, and `split_dataset` documents that grouped data must be split by group.

###### Use cases

The following uses are prohibited even where the model would work:

- classifying people in order to surveil, track, profile or score them, or to infer sensitive attributes;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present labels as verified facts or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed labels;
- any use that violates the Apache-2.0 licence of the converted weights, the MIT licence of the upstream code and of the sample archive, or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `microsoft/cvt-13`
- Revision: `84e365a5f6a5ca987486abb25f3d8e5265cdc44d` (pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest file at it, recorded each file's SHA-256, and recorded the Hub's LFS SHA-256 of the two reference files without downloading them).
- Snapshot manifest: `weights/cvt-13/dimer-base-manifest.json`, 4 staged files, `totalBytes` 80238966, plus two reference files. The byte sizes and digests describe the files at the pinned commit.
- `model.safetensors` (executed artifact): 80,166,694 bytes; SHA-256 `71576c56ac8aaabc1db4e74d86d0316ba1fe7ecff397bbe81b33cf874df8eee2` (matches the Hub's LFS record).
- `pytorch_model.bin` (pickle of the same weights, reference only): 80,260,523 bytes; never staged or loaded; Hub LFS SHA-256 `77c0a16fa66a3c762159966806b8b22e13c4f0e6c879544e83621e6053f5a8e2`.
- `tf_model.h5` (hosted TensorFlow checkpoint, reference only): 80,698,472 bytes; never staged or loaded; Hub LFS SHA-256 `d782bacd92bd57ab1b7e59e5cbff7f1a102e094cc338ac72186af080ebd5abc1`.
- `config.json`: 70,332 bytes; `CvtForImageClassification`, depths 1, 2, 10, widths 64, 192, 384, heads 1, 3, 6, patch sizes 7, 3, 3, 1000 labels.
- `preprocessor_config.json`: 266 bytes; `ConvNextFeatureExtractor`, size 224, `crop_pct` 0.875, bicubic, ImageNet mean and standard deviation.
- `README.md`: 1,674 bytes; the upstream model card.
- Loader: `CvtForImageClassification(CvtConfig.from_pretrained(<verified dir>, local_files_only=True))`, then `load_state_dict(safetensors.torch.load_file(<verified file>), strict=True)`.
- Sample dataset: `Cleanlab/cifar-10-subset` at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`, `CIFAR-10-subset.zip`, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`.

## Input/output contract

- `CvtPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=20260925)`: stage (only with `allow_download=True`), verify, load; with `class_names`, replace the head deterministically under `seed`.
- `predict(images, top_k=None) -> dict`: keys `predictions` (per image: `predicted_label`, `predicted_index`, `top_k` as a list of `{"label", "index", "score"}` in descending score), `top_k`, `decision_rule` (`"argmax"`), `class_names_count`, `adapted`, `device`, `model_id`, `model_revision`. `top_k` defaults to 5, or to the number of classes if smaller.
- `zero_shot_evaluate(records, groups=IMAGENET_GROUPS, *, majority=None) -> dict`: the measures of `evaluate`, plus `rule`, `groups` and `mean_group_mass`; ImageNet head only.
- `finetune(records, *, epochs=5, batch_size=16, learning_rate=1e-4, weight_decay=0.01, seed=20260925, freeze_backbone=False, progress=None) -> dict`: AdamW, cross-entropy, float32, no augmentation; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, majority=None) -> dict`: `accuracy`, `balanced_accuracy`, `per_class_recall`, `support`, `confusion_matrix`, `n`, `mean_top1_score`, `majority_baseline_accuracy` (with `majority`), `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `cvt-adapter-v1`.
- Records: `{"id": str, "image": PIL.Image.Image, "label": str}`; `read_class_archive(zip)` and `read_class_folder(directory)` read them from `<class>/<image>` layouts.
- Constants: `MIN_IMAGE_SIDE = 8`, `MAX_IMAGE_SIDE = 4096`, `MAX_BATCH = 64`, `NUM_IMAGENET_CLASSES = 1000`, `DEFAULT_TOP_K = 5`, `IMAGENET_GROUPS` (frog: 30, 31, 32; truck: 555, 569, 675, 864, 867), `MAX_RECORDS = 5000`, `MIN_PER_CLASS = 2`.

## Verification records

Default-path execution recorded on 2026-09-26 (Kaggle T4): exact notebook blob `5d9e491cd649` at commit `5e0f03b`, 279.5 s, 14/14 post-restart code cells, both BYOD branches off; measured values are under Metrics, and 46 of 60 held-out and 36 of 60 unseen images have a darkened/original counterpart in the training split. REL12 BYOD exercise pending before promotion. The offline test suite runs a narrow `CvtForImageClassification` (1, 2 and 3 blocks) with random weights on 32 px inputs through prediction, the zero-shot baseline, full and frozen fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Wu, Xiao, Codella, Liu, Dai, Yuan and Zhang. CvT: Introducing Convolutions to Vision Transformers. ICCV 2021. https://arxiv.org/abs/2103.15808
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. https://arxiv.org/abs/2010.11929
- Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.
- Upstream code: https://github.com/microsoft/CvT (MIT)
- Upstream card: https://huggingface.co/microsoft/cvt-13
- Sample dataset: https://huggingface.co/datasets/Cleanlab/cifar-10-subset
