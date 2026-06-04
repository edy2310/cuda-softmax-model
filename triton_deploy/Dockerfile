FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /repo

# Copy the project into the image.
COPY . /repo

# Install PyTorch (CUDA) for the Python backend model.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Build the PyTorch CUDA extension.
RUN python3 setup.py build_ext --inplace

# Expose only the HTTP port.
EXPOSE 8000

# Run Triton Inference Server with the model repository.
ENV SOFTMAX_ENGINE_REPO=/repo
CMD ["tritonserver", "--model-repository=/repo/triton_deploy", "--grpc-port=0", "--metrics-port=0"]
