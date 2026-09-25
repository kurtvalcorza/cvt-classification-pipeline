import json
import re
from pathlib import Path

import pytest
from PIL import Image

from cvt_classification_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    IMAGENET_GROUPS,
    MODEL_ID,
    MODEL_KEY,
    is_pinned,
)
from cvt_classification_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "weights" / MODEL_KEY


def test_identity_constants_agree_with_the_committed_manifest():
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert MODEL_ID == "microsoft/cvt-13" == manifest["modelId"]
    assert manifest["revision"] == pipeline_module.MODEL_REVISION
    assert is_pinned() == bool(HEX40.match(pipeline_module.MODEL_REVISION))
    assert DEFAULT_WEIGHTS_DIR == SNAPSHOT and ARTIFACT_FORMAT == "cvt-adapter-v1"
    assert pipeline_module.MODEL_LICENSE == "apache-2.0"
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md", "config.json", "preprocessor_config.json", "model.safetensors",
    }  # fmt: skip
    assert pipeline_module.WEIGHTS_FILE == "model.safetensors"
    assert [r["path"] for r in manifest["referenceFiles"]] == ["pytorch_model.bin", "tf_model.h5"]
    assert all("not loaded" in r["note"] for r in manifest["referenceFiles"])
    if not is_pinned():
        assert all(entry["sha256"] is None for entry in manifest["files"] + manifest["referenceFiles"])


def test_pipeline_transform_matches_the_snapshot_image_processor():
    raw = (SNAPSHOT / "preprocessor_config.json").read_bytes()
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert len(raw) == next(e["bytes"] for e in manifest["files"] if e["path"] == "preprocessor_config.json")
    hub = json.loads(raw)
    assert (hub["size"], hub["crop_pct"], hub["resample"]) == (224, 0.875, 3)
    assert hub["feature_extractor_type"] == "ConvNextFeatureExtractor"
    assert tuple(hub["image_mean"]) == pipeline_module.IMAGENET_DEFAULT_MEAN
    assert tuple(hub["image_std"]) == pipeline_module.IMAGENET_DEFAULT_STD
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    import numpy as np

    processor = transformers.ConvNextImageProcessor.from_pretrained(str(SNAPSHOT))
    for size in ((300, 420), (224, 224), (500, 260)):
        image = Image.fromarray(np.random.default_rng(size[0]).integers(0, 256, (*size, 3), dtype=np.uint8))
        expected = processor(image, return_tensors="pt")["pixel_values"][0]
        assert float((expected - pipeline_module.eval_transform()(image)).abs().max()) < 1e-5


def _labels_with_groups() -> list[str]:
    labels = [f"class {i}" for i in range(1000)]
    for index, name in pipeline_module.IMAGENET_GROUP_LABELS.items():
        labels[index] = f"{name}, extra synonym"
    return labels


def test_imagenet_group_labels_cover_the_groups_and_are_checked():
    assert set(pipeline_module.IMAGENET_GROUP_LABELS) == {i for ids in IMAGENET_GROUPS.values() for i in ids}
    assert 717 not in IMAGENET_GROUPS["truck"]  # CIFAR-10 excludes pickup trucks
    pipeline_module.check_imagenet_groups(_labels_with_groups())
    shifted = _labels_with_groups()
    shifted[30], shifted[31] = shifted[31], shifted[30]
    with pytest.raises(ValueError, match="ImageNet-1k order"):
        pipeline_module.check_imagenet_groups(shifted)
    with pytest.raises(ValueError, match="1000 ImageNet labels"):
        pipeline_module.check_imagenet_groups(["a", "b"])


def test_loader_strict_loads_a_full_size_checkpoint_and_refuses_drift(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file

    model = pipeline_module.build_model()
    assert sum(p.numel() for p in model.parameters()) == 19_997_480  # the upstream README's 20M for CvT-13
    model.config.architectures = ["CvtForImageClassification"]
    model.config.id2label = dict(enumerate(_labels_with_groups()))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.config.save_pretrained(tmp_path)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    loaded = pipeline_module._load_pretrained(tmp_path)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "classifier.bias"}
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="classifier.bias"):
        pipeline_module._load_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["depth"] = [1, 4, 16]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="does not describe CvT-13"):
        pipeline_module._load_pretrained(tmp_path)
