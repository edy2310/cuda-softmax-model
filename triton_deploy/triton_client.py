from __future__ import annotations

import numpy as np
import tritonclient.http as httpclient


def main() -> None:
    url = "localhost:8000"
    batch_size = 4
    features = 561

    x = np.random.randn(batch_size, features).astype(np.float32)

    x_in = httpclient.InferInput("INPUT", x.shape, "FP32")
    x_in.set_data_from_numpy(x)

    client = httpclient.InferenceServerClient(url=url, verbose=False)
    output_req = httpclient.InferRequestedOutput("OUTPUT")
    result = client.infer(
        "softmax_engine_repository",
        inputs=[x_in],
        outputs=[output_req],
    )
    probs = result.as_numpy("OUTPUT")
    print(f"Output shape: {probs.shape}, dtype: {probs.dtype}")


if __name__ == "__main__":
    main()
