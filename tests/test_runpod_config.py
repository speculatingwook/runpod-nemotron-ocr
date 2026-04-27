import json
from pathlib import Path


def test_hub_config_requires_cuda_13_runtime_for_ngc_2509_base():
    hub_config = json.loads(Path(".runpod/hub.json").read_text())

    assert hub_config["config"]["allowedCudaVersions"] == ["13.0"]


def test_hub_release_test_requires_cuda_13_runtime_for_ngc_2509_base():
    tests_config = json.loads(Path(".runpod/tests.json").read_text())

    assert tests_config["config"]["allowedCudaVersions"] == ["13.0"]
