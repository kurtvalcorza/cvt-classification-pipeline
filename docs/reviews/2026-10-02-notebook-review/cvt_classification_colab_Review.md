# CvT-13 Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/cvt-classification-pipeline`  
**Notebook:** `tutorials/cvt_classification_colab.ipynb`  
**Reviewed commit:** `3b30a2889a98f1b0065b96653914941a5547418d` (`main`, confirmed with `gh api repos/kurtvalcorza/cvt-classification-pipeline/commits/main` before and after the probes)  
**Notebook Git blob:** `5d9e491cd649f03ea7551a307f68e0f1290237bf`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-26 (commit `5e0f03b`, `fetched_blob_verified: true`); `git diff --stat 5e0f03b 3b30a28` touches only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md`. Generator `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CVT`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

This notebook carries both package modules byte-for-byte (`data.py` 341 lines, `pipeline.py` 868 lines; cell metadata digests equal the module digests), stages and re-hashes the pinned `microsoft/cvt-13` snapshot, downloads a digest-checked CIFAR-10 subset, validates it, splits it three ways, measures majority-class, zero-shot ImageNet-mapping and untrained-head baselines, fine-tunes with every hyperparameter a form field, prints the trainable-parameter count, scores held-out and unseen splits with accuracy, balanced accuracy and a confusion matrix, probes blank and noise images before and after adaptation, exports an adapter with base-model provenance, and checks the reloaded adapter against the in-memory model with a stated tolerance. The prose is careful about what the loss, the scores and a green run do and do not establish.

A direct CPU run of every code cell at the documented defaults reproduced the Kaggle record closely:

| Measure | This review (CPU, pinned venv, defaults) | Kaggle T4 record (blob `5d9e491c`) |
|---|---|---|
| Code cells completed | 14/14 (135.3 s of cell time; install skipped) | 14/14 on pass 2 (pass 1 stopped at the install guard) |
| Dataset | 400 records, 200 per class, 32×32, **100 pixel-identical groups** (finding printed) | identical |
| Split | 280 / 60 / 60 | identical |
| Held-out accuracy: majority / zero-shot / untrained head / fine-tuned | 0.500 / 0.983 / 0.250 / 1.000 | 0.500 / 0.9833 / 0.250 / 1.000 |
| Unseen accuracy, fine-tuned | 1.000 | 0.9833 |
| Held-out / unseen images with a pixel copy in train | **25/60 / 19/60** | not measured (record counts 46/60 / 36/60 same-numbered counterparts) |
| Fine-tune cell | 114.2 s (19,613,250 trainable) | about 20 s |
| Adapted head, blank / noise | `frog` 0.686 / `frog` 0.629 | `frog` 0.546 / `frog` 0.683 |
| Reload check | 60 images, tolerance 1e-4, equivalent | identical |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CVT-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart.
2. **The split ignores the duplicates the notebook itself reports (CVT-M2).** `validate_dataset` prints "100 group(s) of pixel-identical images" (each `darkened_images/<class>/image_N.png` equals `original_images/<class>/image_N.png`), then Section 5 says a random split "is valid here because CIFAR-10 images are independent thumbnails". 25 of 60 held-out and 19 of 60 unseen images have an exact copy in the training split.
3. **The suggested head-only experiment silently runs on the already fine-tuned model when only the edited cell is re-run (CVT-M3).** It reports 770 trainable parameters and 1.000 held-out accuracy, and the export cell then fails with a bare `AssertionError`. Nothing says which cells to re-run; re-running from cell 17 gives the real head-only result, 0.900.
4. **Guided layer mostly absent (CVT-M4).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, task contract, glossary, prediction, checkpoint, troubleshooting or conclusion template, and the 1,209 lines of carried modules are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.1** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Knowledge prerequisites: basic Python and PIL; softmax; image transformers on tokens; accuracy, balanced accuracy, confusion matrix; why a baseline is needed |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA GPU (T4) documented for the fine-tune, CPU also runs; float32 |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-verified CIFAR-10 subset (`frog`, `truck`) validated before any model runs; stratified three-way split; majority-class and zero-shot baselines; blank/noise probes; head replacement and untrained baseline; bounded fine-tune; held-out comparison; unseen-split inference; adapter export, fresh reload and equivalence check; outputs with provenance; BYOD image and BYOD dataset branches, the latter through "the full adaptation workflow — validate, split, baselines, fine-tune, evaluate, export and reload" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `9a914ff` |

### Evidence actually obtained

- **Source inspection.** All 31 cells (14 code; cells 5 and 7 carry `data.py` and `pipeline.py`). Also read: `split_dataset`, `validate_dataset`, `finetune`, `save_artifact`, `load_artifact` in the modules; the generator and template; `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `dimer-nb2-cvt-classification` v1 (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`). Kaggle Tesla T4, 2026-09-26, **the reviewed blob**, clean model cache, image torch 2.10.0+cu128 / numpy 2.0.2. Pass 1 failed in cell 3 with the restart `RuntimeError` (203.4 s); pass 2 ran 14/14 (76.0 s). No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, build venv `dimer-next16` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, transformers 4.57.6, safetensors 0.8.0, numpy 2.5.3, pillow 11.3.0, huggingface-hub 0.36.2 — the notebook's `PINS`). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Not a clean runtime:** the four snapshot data files were hard-linked into a scratch working directory; cell 9 wrote the manifest fresh and `verify_snapshot` re-hashed every file. Cell 11 fetched the dataset itself from its pinned URL and checked its digest.
  - **Executed:** every code cell at defaults (P1); duplicate leakage across the split and scores on the leak-free subsets (P2); untrained head at init seeds 0–4 (P3); "Try next" head-only experiment re-running only cell 19 (P4) and re-running cells 17–25 (P5); "Try next" `PER_CLASS = 10` re-running cells 11–21 (P6); BYOD image through `BYOD_IMAGE_PATH` with one valid and three invalid inputs, an empty path outside Colab, and an upload shim (P7); BYOD dataset through `BYOD_DATASET_PATH` with three accepted and six refused inputs (P8). One invocation, 253.8 s wall. `google.colab.files.upload` was replaced by a shim for one upload probe; the Colab upload dialog itself was not exercised.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good on the default path (P1 14/14; reload equivalent within 1e-4 on 60 images; most BYOD refusals name the failed rule). Defects: the install pattern forces a restart (CVT-M1); a partial re-run of the fine-tune cell leaves the pipeline in a state the export cannot represent (CVT-M3).
- **Promise fulfilment:** all default-path stages run and are reported. The BYOD dataset branch reaches fine-tune and export but not new-data inference, an equivalence check or a written record (CVT-m1).
- **Scientific validity:** the split ignores 100 duplicate pairs the validator reports, and the prose asserts an independence the data lacks (CVT-M2). On this class pair the leak does not move the numbers much (leak-free held-out: fine-tuned 35/35, zero-shot 34/35), because zero-shot is already near ceiling. The fine-tune's measured gain over zero-shot is one image and is not stated as such in the Interpretation section (CVT-m2).
- **Learner experience:** clear stage prose, two "What to look for" notes, an explicit "read the loss as optimisation evidence only" note and an honest limits section. The untrained-head expectation does not match the default number (CVT-m2), re-run guidance is missing (CVT-M3), and the GDL layer is mostly absent (CVT-M4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6 (CVT-M1); SPL3, SPL5 (CVT-M2); DAT14, VER5, DAT19 (CVT-m1); UX12 (CVT-m3); REL12 BYOD evidence absent from the release record. SHOULD deviations: SPL10 (CVT-M2); GDL10, UX10 (CVT-M3); GDL1–GDL4, GDL6, GDL9, GDL11–GDL13, UX8 (CVT-M4); SPL2 (CVT-m1); GDL7, GDL8, GDL14 (CVT-m2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 14/14 | Section 1 says the cell "stops with a restart instruction" | **Not met** (CVT-M1) |
| Digest-verified pinned snapshot | cell 9 | 4/4 files verified at `84e365a5…` | clear | Met |
| Digest-verified dataset, validated before any model | cell 11 | 986,707 B, sha256 `66f90a4f…`; 100 duplicate groups reported | "What to look for" explains 32 px upsampling; duplicate finding not discussed | Met (finding ignored downstream, CVT-M2) |
| Stratified split; "valid here because … independent thumbnails" | cell 13, `split_dataset` | 280/60/60, ID-leakage assert passes; 25/60 held-out and 19/60 unseen pixel copies in train | independence asserted | **Not met** (CVT-M2) |
| Majority and zero-shot baselines | cell 13 | 0.500, 0.983 | zero-shot explained | Met |
| Blank/noise probes before and after | cells 15, 23 | ImageNet head flat (top score ≤ 0.016); adapted head `frog` 0.686 / 0.629 (CPU), 0.546 / 0.683 (T4) | well explained | Met |
| Untrained head "Expect roughly chance" | cell 17 | 0.250 at seed 0 (Wilson 95 % [0.16, 0.37], excludes 0.5); 0.25–0.48 over seeds 0–4 | expectation not matched at the default seed, no comment | Partly met (CVT-m2) |
| Bounded fine-tune, trainable set and schedule stated | cell 19 | 19,613,250 / 19,613,250 trainable; losses printed | clear | Met |
| Held-out comparison against both baselines | cell 21 | fine-tuned 1.000 vs zero-shot 0.983 (one image) | "a difference of one or two images is within noise"; Interpretation does not state this run's result | Computation met, conclusion left implicit (CVT-m2) |
| Unseen-split inference | cell 23 | 1.000 (CPU), 0.9833 (T4) | — | Met (leak caveat, CVT-M2) |
| Export, fresh reload, equivalence with tolerance | cell 25 | 459 tensors; 60 images equivalent within 1e-4 | "loading succeeding is not the check" | Met on the default path; breaks after a partial re-run (CVT-M3) |
| Outputs with provenance | cell 27 | 5 files; result JSON holds dataset digest, split, fine-tune config, metrics, artifact descriptor, reload check | listed | Met |
| BYOD image | cell 29 | P7: path read, top-5 + `not-measurable`; 5000 px and missing path refused with the rule; text file → `UnidentifiedImageError` naming the path; empty path outside Colab → `ModuleNotFoundError: google.colab` | contract stated | Met (local), one weak message (CVT-m1) |
| BYOD dataset "full adaptation workflow" | cell 29 | P8: folder and zip reach validate → split → fine-tune → export → load; no unseen split, no new-data inference, no equivalence check, no metrics written | contract stated | Partly met (CVT-m1) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried package, verify the revision | run cells | versions and verified-file count printed |
| Download and validate a dataset before any model runs | run cell, read manifest | duplicate finding printed, then ignored (CVT-M2) |
| Read top-5; build a zero-shot baseline | run, read | outputs readable |
| See what a closed-set classifier answers for blank and noise | run, read | "What to look for" note; strong |
| Split with stratification | run | split is not duplicate-aware (CVT-M2) |
| Fine-tune; compare against both baselines | run, read table | no prompt to interpret a one-image difference (CVT-m2) |
| Export, reload and verify the adapter | run | met |

Objectives are phrased as actions the code performs (GDL5), and none is followed by a check of the learner's understanding.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | Each section says what runs and why; two "What to look for" notes; the loss and the held-out metrics are labelled as optimisation and tutorial evidence. The learner is told the split is valid because the thumbnails are independent right after being shown 100 duplicate groups (CVT-M2); is told to expect chance from the untrained head and sees 0.250 with no comment (CVT-m2). Prerequisites say "Runtimes are not measured in this revision" although the record measured them (CVT-m3). No audience, roadmap, glossary, predictions, checkpoints or troubleshooting (CVT-M4). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard after 203.4 s, pass 2 14/14 after a restart (CVT-M1). Direct: 14/14 at defaults, numbers in the table above; five outputs written; reload equivalent. No Colab run. |
| **Active learning** | Direct (P4, P5, P6) | "Try next" 1, as a learner would do it — set `FREEZE_BACKBONE = True` in cell 19 and re-run that cell: training continues from the fully fine-tuned weights, the cell prints `trainable_parameters: 770`, cell 21 shows 1.000, and cell 25 raises a bare `AssertionError` because the head-only adapter cannot reproduce a modified backbone (CVT-M3). Re-running cells 17–25 instead works: head-only 0.900 held-out (frog recall 0.833, truck 0.967) in 33.9 s vs full 1.000 in 114.2 s (CPU), 2-tensor adapter, reload equivalent. "Try next" 2, `PER_CLASS = 10` with cells 11–21 re-run: split 14/4/2, zero-shot 1.000, untrained 0.500, fine-tuned 0.750 on 4 held-out images — the fine-tune falls below zero-shot, on a split too small to read. |
| **Reuse and recovery** | Direct (P7, P8); Colab upload dialog not verified | BYOD image through `BYOD_IMAGE_PATH`: accepted and classified (`leopard` 0.162 on an upscaled CIFAR frog), report `not-measurable`; refused with the rule named: 5000×10 px image, missing file; raw `UnidentifiedImageError` (path named) for a text file named `.png`; empty path on a non-Colab runtime → `ModuleNotFoundError: No module named 'google.colab'`. BYOD dataset (stand-in 2×6 synthetic stripe images): folder and zip reached validate → split (8/4) → untrained 0.75 → fine-tune → 0.50 → export → load. Refused with the rule named: class with one image, one class, `../` member, undecodable member (named). Accepted silently: a `train/` + `val/` archive (merged by class and re-split). Weak messages: root-level images ignored, then "class_names must hold 2..1000 names, got 1"; a `.tar` path → "is not a directory" (CVT-m1). |

## 5. Findings

### Major

#### CVT-M1 — `Run all` needs a manual restart after the install cell

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, install block lines 50–75, `RuntimeError` at line 72); `docs/release-verification.md` record.
- **Observed issue:** the cell `pip install`s seven pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. Section 1 prose presents this as expected behaviour.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle/Colab image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this. The record reports the run as "PASSED (default path)", with the restart disclosed in the same cell and described as "as the notebook instructs".
- **Evidence:** documented — Kaggle T4 run of blob `5d9e491c`, pass 1 `ok: false` (203.4 s) with `cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3`, `restarted_after_install_cell: true`, pass 2 14/14 (76.0 s). Source — probe static `pip_install_in_kernel: true`, `uses_uv: false`.
- **Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/cvt_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### CVT-M2 — The split ignores the 100 duplicate pairs the validator reports, and the prose asserts independence

- **Cell/section:** cell 11 output (`duplicate_groups: 100`), Section 5 prose ("A random split is valid here because CIFAR-10 images are independent thumbnails"), cell 13; `data.py` `split_dataset`. Generator: `tools/notebook_template.py` lines 152–155 and 169–170; `src/cvt_classification_pipeline/data.py` `split_dataset` (line 308).
- **Observed issue:** the archive holds `original_images/<class>/image_N.png` and `darkened_images/<class>/image_N.png`; for 100 values of N the two files are pixel-identical. `validate_dataset` reports this as a finding, and its own docstring says a duplicate "inflates the held-out score when one copy lands on each side of a split". `split_dataset` then shuffles individual records; the only leakage check (cell 13) compares record IDs, which always differ between the two copies.
- **Consequence:** 25 of the 60 held-out and 19 of the 60 unseen images have an exact copy in training, so the "held-out" and "unseen" scores are partly scores on training images, and the learner is taught that a random split is valid on data the notebook has just flagged as duplicated. On this class pair the numbers barely move (leak-free held-out 35 images: fine-tuned 1.000, zero-shot 0.971; leak-free unseen 41 images: fine-tuned 1.000), but the method carries straight into the BYOD branch, where it would inflate a user's score silently.
- **Evidence:** direct (P2) — `held_out_with_pixel_copy_in_train: 25/60`, `unseen_with_pixel_copy_in_train: 19/60`, all groups of size 2, no group spans two labels; documented — the record's caveat (counts 46/60 and 36/60 same-numbered counterparts, not pixel copies). Source — `split_dataset` has no group parameter.
- **Recommended correction:** make the split group-aware: group records by pixel digest (or by `image_N` across the two folders) and assign whole groups to one side, then assert in cell 13 that no pixel digest appears in two splits and print the count. Alternatively drop one of each duplicate pair before splitting and say so. Replace the "independent thumbnails" sentence with the actual reason the split is valid after de-duplication, and apply the same rule in the BYOD branch.
- **Acceptance check:** cell 13 prints a cross-split duplicate count of 0 for held-out and unseen, computed from pixel digests; the Section 5 prose no longer claims the archive is independent; a test in `tests/test_data.py` builds an archive with a duplicate pair and shows both copies land in one split.
- **Spec:** SPL3, SPL5, SPL10, EVAL5.

#### CVT-M3 — Re-running only the edited fine-tune cell trains the already adapted model, misreports it, and breaks the export

- **Cell/section:** cell 19 (`FREEZE_BACKBONE` form field), cells 21 and 25, "Try next" in the Interpretation section; `pipeline.py` `finetune` (mutates `self`), `save_artifact` (line 768, omits `frozen_prefixes` tensors). Generator: `tools/notebook_template.py` lines 218, 240 and 458.
- **Observed issue:** `adapter` is created in cell 17 and mutated in place by `finetune`. A learner following "Set `FREEZE_BACKBONE = True` and compare" naturally edits cell 19 and re-runs it. That continues training the fully fine-tuned network with its backbone frozen, prints `trainable_parameters: 770`, and cell 21 reports 1.000 as if it were a head-only result. Cell 25 then saves only the head (the backbone is "frozen", so assumed equal to the base), reloads it onto the base model, and the predictions differ: bare `AssertionError` with no message. No cell says which cells to re-run after changing a field.
- **Consequence:** the notebook's one guided experiment produces a wrong comparison (full + head-only training labelled head-only: 1.000 instead of the true head-only 0.900) and then crashes without a recovery message. The correct procedure (re-run from cell 17) works, but the learner is not told it.
- **Evidence:** direct (P4) — re-run of cell 19 only: losses 0.2010 → 0.1441, `trainable_parameters 770`, held-out 1.000, cell 25 `AssertionError:`; (P5) re-run of cells 17–25: 770 trainable, held-out 0.900, 2-tensor adapter, reload equivalent, fine-tune 33.9 s vs 114.2 s full on CPU.
- **Recommended correction:** make the experiment self-contained: build a fresh pipeline inside the fine-tune cell (or refuse to fine-tune a pipeline whose `adapted` is already `True` with a message saying "re-run from Section 7"), print in the "Try next" text exactly which cells to re-run, and give the reload assertions messages that name the mismatch and the next step.
- **Acceptance check:** with defaults run once, setting `FREEZE_BACKBONE = True` and re-running only cell 19 either trains from the base checkpoint (cell 25 then passes) or stops with a message naming the cells to re-run; every "Try next" item names its field and the cells to re-run.
- **Spec:** GDL10, UX10, UX5, FT5.

#### CVT-M4 — Declared `GUIDED`, but most of the guided layer is absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9, end of notebook. Generator: `tools/notebook_template.py`, `tools/build_notebook.py`.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output contract (ImageNet classification and two-class adaptation share one notebook), no glossary (convolutional token embedding, convolutional projection, BatchNorm running statistics, softmax, balanced accuracy, AdamW, cross-entropy, adapter, zero-shot mapping), no prediction prompts before the zero-shot, untrained-head or fine-tune results, no interpretation checkpoints, no troubleshooting section (restart, Hub download, CPU time, BYOD errors), no conclusion template. The two module cells (1,209 lines) and the manifest cell are not titled **Infrastructure** and are not collapsed (`cellView` absent everywhere). The notebook does have two "What to look for" notes, expectation notes in Sections 7 and 8, and a strong limits section.
- **Consequence:** a self-paced learner gets an accurate, well-explained script but no prompts to commit to a prediction, check understanding or recover from expected failures, and must scroll past 1,209 lines of package code without being told it can be skipped.
- **Evidence:** source inspection; probe static `guided_markers` (How to use / Roadmap / Glossary / Check your reasoning / Troubleshooting / conclusion / audience / Infrastructure / re-run all absent), `cellView_form_cells: []`, `what_to_look_for_count: 2`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC §25.13's reference notebook: audience and how-to-use, roadmap, task contracts for both capabilities, glossary, a prediction before Sections 5, 7 and 9, a collapsible checkpoint after the comparison table, a Predict → Change one thing → Run → Observe → Explain activity built on CVT-M3's fix, troubleshooting, and a conclusion scaffold; title cells 3, 5, 7 and 9 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`; cells 3, 5, 7 and 9 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL4, GDL6, GDL9, GDL11–GDL14, UX8.

### Minor

#### CVT-m1 — BYOD dataset branch stops short of the promised workflow; some inputs get weak messages

- **Cell/section:** cell 29; opening cell and Section 13 prose ("the same validate → split → baselines → fine-tune → evaluate → export → reload stages as the sample"). Generator: `tools/notebook_template.py` lines 379–438.
- **Observed issue:** the branch splits train/held-out only (no unseen split or new-data inference), calls `load_artifact` without comparing the reloaded predictions (the default path does compare), writes no metrics, dataset manifest or split record (only the adapter file), and prints `{'majority_baseline', 'untrained', 'fine_tuned'}` mixing accuracy and balanced accuracy without saying so. A `train/` + `val/` archive is merged by class name and re-split silently. Images at the archive root are ignored without a message, so an archive with one class folder plus root images fails with "class_names must hold 2..1000 names, got 1". A `.tar` path fails as "is not a directory". With an empty path on Kaggle or local Jupyter the branch raises `ModuleNotFoundError: No module named 'google.colab'`. The BYOD branch also inherits CVT-M2's split.
- **Consequence:** a user's adaptation result is printed once and not recorded; reload is a load, not a check; pre-split data loses its split; some failures do not tell the user what to fix.
- **Evidence:** direct (P8, stand-in synthetic images) — `a_folder_ok`/`b_zip_ok`: split 8/4, untrained 0.75, fine-tuned 0.50, adapter exported and loaded, nothing else written to `outputs/`; `g_presplit_train_val` accepted, 12 records re-split 8/4; `h_root_images_plus_one_class` → `ValueError: class_names must hold 2..1000 names, got 1`; `i_tar_path` → `FileNotFoundError: … is not a directory`. (P7) empty path outside Colab → `ModuleNotFoundError`. Refused with a clear rule: class with one image, one class, `../` member, undecodable member (named), 5000 px image, missing file.
- **Recommended correction:** reuse the default path's reload comparison and output writer for BYOD (`outputs/byod_result.json` with dataset manifest, split, fine-tune config and metrics); add the unseen/new-data split and its predictions; label the printed metrics; keep a `train/`/`val/` layout as the split when present (or say it is merged); report ignored root-level files; when the path is empty and `google.colab` is unavailable, stop with "set `BYOD_DATASET_PATH`"; state the `.zip`-or-directory rule in the error.
- **Acceptance check:** a BYOD dataset run writes a result JSON with metrics, new-data predictions and a reload check marked `equivalent`; the root-images, `.tar` and empty-path-outside-Colab cases each stop with a message naming the rule and the field to set.
- **Spec:** DAT14, VER5, DAT19, SPL2, UX10, REL12.

#### CVT-m2 — The untrained-head expectation and the fine-tune result are not interpreted against what the run shows

- **Cell/section:** Section 7 prose ("**Expect roughly chance.** … This row shows the floor"), cell 21, Interpretation section and "Try next". Generator: `tools/notebook_template.py` line 214 and lines 443–460.
- **Observed issue:** the untrained head scores 0.250 at the default `SEED = 0` (Wilson 95 % [0.16, 0.37], below the 0.5 majority baseline); across seeds 0–4 it scores 0.25–0.48. The prose calls this "roughly chance" without noting that a random head can land well below chance. The fine-tune beats zero-shot by one held-out image (0.983 → 1.000; leak-free 34/35 → 35/35); the Section 9 note says such a difference is within noise, but the Interpretation section only says the result "was compared with" the baselines and never states that this run measured no reliable gain over zero-shot. "Try next" asks the learner to "watch how quickly the fine-tuned model falls back toward the zero-shot baseline", which presupposes an outcome; at `PER_CLASS = 10` the held-out split has 4 images.
- **Consequence:** the learner can leave believing the fine-tune clearly improved on the pretrained model, and the first baseline number contradicts the stated expectation without explanation.
- **Evidence:** documented — Kaggle held-out 0.500 / 0.9833 / 0.250 / 1.000 and the record's own "a one-image difference within the noise of this split"; direct — P1 identical; P2 leak-free subsets; P3 seeds; P6 `PER_CLASS = 10`: zero-shot 1.000, fine-tuned 0.750 on 4 images.
- **Recommended correction:** replace "Expect roughly chance" with "a random head carries no information about the task and can land well below or above 50 %"; add one sentence after cell 21 and in the Interpretation section stating this run's fine-tuned vs zero-shot result and that a one-image difference is not evidence of a gain; reword the `PER_CLASS` experiment as a question ("does the fine-tune still beat zero-shot?") and warn that the held-out split shrinks with it.
- **Acceptance check:** `grep -n "Expect roughly chance" tutorials/cvt_classification_colab.ipynb` returns nothing; the Interpretation section names the fine-tuned vs zero-shot result of the run; the `PER_CLASS` item states the held-out size it produces.
- **Spec:** EVAL3, GDL7, GDL8, GDL14.

#### CVT-m3 — Runtime claim and release record out of step with the evidence

- **Cell/section:** cell 1 Prerequisites ("Runtimes are not measured in this revision"); `docs/release-verification.md` caveat on duplicates.
- **Observed issue:** the record measured 279.5 s wall on T4 (pass 2 76.0 s, fine-tune about 20 s) for this exact blob; the notebook still says runtimes are not measured and gives no CPU estimate although it says CPU works "more slowly" (this review: fine-tune 114.2 s, all cells 135.3 s on a 24-thread CPU). The record's leakage caveat counts 46/60 and 36/60 same-numbered counterparts; the pixel-copy counts are 25/60 and 19/60.
- **Consequence:** the learner cannot plan time; a release reviewer reads a leakage figure that overstates the measured one (the record says it is a counterpart count, but does not give the measured figure).
- **Evidence:** documented record; direct P1 and P2 timings and counts.
- **Recommended correction:** state measured times with runtime and date (T4; CPU labelled as an estimate) in the Prerequisites; replace the counterpart count in the record with the pixel-digest count (or remove it once CVT-M2 is fixed).
- **Acceptance check:** the Prerequisites give a measured T4 time with date and runtime; the record's duplicate figure is computed from pixel digests.
- **Spec:** UX12, REL10.

### Suggestions

- **CVT-S1** — Declare `notebook_spec` 2.2 instead of 2.1 once the guided layer lands.
- **CVT-S2** — Show a small grid of held-out images with true label, predicted label and score, including the zero-shot error, next to the confusion matrix.
- **CVT-S3** — Report the untrained head as a band over a few init seeds (P3 shows 0.25–0.48) so the "floor" is not read from one draw.
- **CVT-S4** — Record per-stage wall times in `result.json` (the fine-tune dominates: 114.2 of 135.3 s on CPU).

## 6. Readiness

**Needs revision.** Open Majors CVT-M1 to CVT-M4. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10), the REL12 BYOD exercise recorded in `docs/release-verification.md` (one compatible and one incompatible input), and a default run whose held-out and unseen splits contain no pixel copy of a training image.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, labelled above):** default path 14/14 with the T4 record's held-out metrics; 25/60 held-out and 19/60 unseen pixel copies in train and the scores on the leak-free subsets; untrained-head spread across seeds; the partial re-run failure and the working full re-run (head-only 0.900); the `PER_CLASS = 10` outcome; the BYOD image and dataset outcomes in §4 (dataset outcomes on synthetic stand-in images).
- **Verified from documented evidence:** the restart on Kaggle pass 1; the T4 metrics and timings.
- **Inferred from source:** that the Colab upload dialog delivers files as the shim did; that GPU runs show the same partial-re-run failure (it follows from `finetune` mutating `self` and `save_artifact` omitting frozen tensors).
- **Not verified:** any Colab run; BYOD on real user photographs.
- **Most likely to be wrong:** CVT-M2's severity — on this data the leak barely changes the numbers (leak-free held-out 35/35 fine-tuned, 34/35 zero-shot), so a maintainer could rate it Minor; I rated it Major because the notebook teaches a split rule its own validator contradicts, and the BYOD branch applies it to user data.
