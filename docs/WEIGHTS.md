# Weight and sample-data provenance and hosting

- Upstream: `microsoft/cvt-13`
- Revision: `84e365a5f6a5ca987486abb25f3d8e5265cdc44d`, pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest-listed file at it, cross-checked each LFS file against the SHA-256 the Hub records, recorded the Hub's LFS SHA-256 of the two reference files, and wrote the commit and digests into the manifest and `src/cvt_classification_pipeline/pipeline.py`.
- Executed artifact: `model.safetensors` (80,166,694 bytes, SHA-256 `71576c56ac8aaabc1db4e74d86d0316ba1fe7ecff397bbe81b33cf874df8eee2`).
- Hosted, not executed: `pytorch_model.bin` (80,260,523 bytes), a pickle of the same weights, and `tf_model.h5` (80,698,472 bytes), the TensorFlow checkpoint. Both are listed under `referenceFiles`; the pin tool records their Hub LFS SHA-256 without downloading them.
- Manifest: `weights/cvt-13/dimer-base-manifest.json` (4 staged files: `README.md`, `config.json`, `preprocessor_config.json`, `model.safetensors`; `totalBytes` 80238966; plus the two reference files). Every entry carries its SHA-256.
- Committed copy: `preprocessor_config.json` (266 bytes) is committed as the Hub serves it at the pinned commit, so a test can check the pipeline's torchvision transform against the Transformers processor built from it. The pin tool replaced it with the bytes downloaded at the pinned commit before hashing (they were byte-identical), so its digest describes the pinned bytes.
- Upstream weight licence: Apache-2.0 (the checkpoint's `README.md` front matter). The upstream `microsoft/CvT` repository, which released the original weights, is MIT-licensed; both licences are permissive. The model was trained on ImageNet-1k, whose images carry their own terms of access.
- Hosting: the Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored), and nothing here redistributes the weights.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: `CvtConfig.from_pretrained(<verified dir>, local_files_only=True)` and `CvtForImageClassification(config)` build the architecture without contacting the Hub or running repository code; `load_state_dict(..., strict=True)` loads the verified SafeTensors file and refuses a missing, unexpected or mis-shaped tensor; and the loader refuses a config whose labels at the zero-shot group indices are not the expected ImageNet classes.

## Tutorial sample data

- Dataset: `Cleanlab/cifar-10-subset`, file `CIFAR-10-subset.zip`, at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`.
- Size and digest: 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`. `fetch_sample_archive` refuses any other bytes and has no fallback.
- Licence: MIT (the dataset card). The images are CIFAR-10 images (Krizhevsky, 2009); the tutorial keeps the `frog` and `truck` folders.
- Hosting: the archive is downloaded at runtime into a working directory (`data/`, git-ignored) and is not redistributed by this repository.
