import os
import sys
import urllib.request
import zipfile

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core.registered import DatasetNotFoundError
import torch

# Ensure the project root (where the .so is built) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import softmax_warp_ext

from models.mlp import HARMLP


def download_uci_har(data_dir: str) -> str:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
        "UCI%20HAR%20Dataset.zip"
    )
    zip_path = os.path.join(data_dir, "UCI_HAR_Dataset.zip")
    dataset_root = os.path.join(data_dir, "UCI HAR Dataset")

    os.makedirs(data_dir, exist_ok=True)
    if os.path.isdir(dataset_root):
        return dataset_root

    print("TFDS dataset not available. Downloading UCI HAR from UCI repository...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    return dataset_root


# Load a fixed pool of samples from UCI HAR (train split)
def load_uci_har_samples(max_samples=64):
    data_dir = "data"
    try:
        ds_train = tfds.load(
            "uci_har",
            data_dir=data_dir,
            split="train",
            as_supervised=True,
        )

        xs = []
        for x, _ in tfds.as_numpy(ds_train):
            xs.append(x)
            if len(xs) == max_samples:
                break

        return np.asarray(xs, dtype=np.float32)
    except DatasetNotFoundError:
        dataset_root = download_uci_har(data_dir)
        x_train = np.loadtxt(
            os.path.join(dataset_root, "train", "X_train.txt"),
            dtype=np.float32,
        )
        return x_train[:max_samples]


def assert_raises(fn, expected_substr):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - intentional for simple test
        if expected_substr not in str(exc):
            raise AssertionError(
                f"Expected error containing '{expected_substr}', got '{exc}'"
            ) from exc
        return
    raise AssertionError("Expected an exception but none was raised")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this test.")

    # Create two models with identical weights
    torch.manual_seed(0)
    model_pt = HARMLP().cuda().eval()
    model_custom = HARMLP(softmax_fn=softmax_warp_ext.forward).cuda().eval()
    model_custom.load_state_dict(model_pt.state_dict())

    # ----- Happy path tests (multiple batch sizes) -----
    samples = load_uci_har_samples(max_samples=64)
    kernel_name = "softmax_warp_ext.forward"
    for batch_size in [1, 8, 32]:
        x = torch.tensor(samples[:batch_size], device="cuda")
        with torch.no_grad():
            out_pt = model_pt(x)
            out_custom = model_custom(x)

        if not torch.allclose(out_pt, out_custom, rtol=1e-4, atol=1e-5):
            max_diff = (out_pt - out_custom).abs().max().item()
            raise AssertionError(
                "FAIL: Outputs differ "
                f"(test=equivalence batch {batch_size}, kernel={kernel_name}, "
                f"max diff = {max_diff})"
            )

    print("PASS: Outputs match for batch sizes 1, 8, 32.")

    # ----- Bad path tests (input validation in the binding) -----
    assert_raises(
        lambda: softmax_warp_ext.forward(torch.randn(4, 6)), "input must be a CUDA tensor"
    )
    assert_raises(
        lambda: softmax_warp_ext.forward(
            torch.randn(4, 6, device="cuda", dtype=torch.float64)
        ),
        "input must be float32",
    )
    assert_raises(
        lambda: softmax_warp_ext.forward(torch.randn(2, 3, 4, device="cuda")),
        "input must be 2D",
    )

    print("PASS: Bad path validations triggered as expected.")


if __name__ == "__main__":
    main()
