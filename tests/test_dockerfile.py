from pathlib import Path


def test_dockerfile_installs_nemotron_package_from_repo_subdirectory():
    dockerfile = Path("Dockerfile").read_text()

    assert "cd /opt/nemotron-ocr-v2/nemotron-ocr" in dockerfile
    assert "pip install --no-build-isolation -v ." in dockerfile
