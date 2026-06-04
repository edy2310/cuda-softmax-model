from __future__ import annotations

from pathlib import Path
import os
import sys

import torch
import triton_python_backend_utils as pb_utils


def _add_repo_to_path() -> None:
    # Allow overriding the repo root with an env var for deployment flexibility.
    repo_root = os.environ.get("SOFTMAX_ENGINE_REPO")
    if repo_root is None:
        # triton_deploy/softmax_engine_repository/1/model.py -> repo root is 3 levels up.
        repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class TritonPythonModel:
    def initialize(self, args):
        _add_repo_to_path()
        from models.mlp import HARMLP
        import softmax_warp_ext

        self.device = torch.device("cuda")
        self.model = HARMLP(softmax_fn=softmax_warp_ext.forward).to(self.device).eval()

    def execute(self, requests):
        responses = []
        with torch.no_grad():
            for request in requests:
                x_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
                x = torch.from_numpy(x_tensor.as_numpy()).to(self.device)
                if x.dtype != torch.float32:
                    x = x.float()

                probs = self.model(x)
                out = pb_utils.Tensor("OUTPUT", probs.cpu().numpy())
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
