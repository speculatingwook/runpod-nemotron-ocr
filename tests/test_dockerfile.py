from pathlib import Path


def test_dockerfile_installs_nemotron_package_from_repo_subdirectory():
    dockerfile = Path("Dockerfile").read_text()

    assert "cd /opt/nemotron-ocr-v2/nemotron-ocr" in dockerfile
    assert "pip install --no-build-isolation -v ." in dockerfile


def test_dockerfile_installs_nemotron_build_backend_before_no_isolation_install():
    dockerfile = Path("Dockerfile").read_text()
    backend_install = "pip install --no-cache-dir hatchling editables"
    package_install = "pip install --no-build-isolation -v ."

    assert backend_install in dockerfile
    assert dockerfile.index(backend_install) < dockerfile.index(package_install)


def test_dockerfile_targets_a100_and_ada_cuda_architectures():
    dockerfile = Path("Dockerfile").read_text()

    assert 'ENV TORCH_CUDA_ARCH_LIST="8.0;8.9+PTX"' in dockerfile
