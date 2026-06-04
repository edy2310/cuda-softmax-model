import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mlp import HARMLP


def main():
    # Default values
    data_dir = "data"
    epochs = 10
    batch_size = 64
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Show which device will be used
    print(f"Using device: {device}")

    # Load UCI HAR from TensorFlow Datasets (train/test splits already defined)
    ds_train, ds_test = tfds.load(
        "uci_har",
        data_dir=data_dir,
        split=["train", "test"],
        as_supervised=True,
    )

    # Convert TF datasets to numpy arrays and then to PyTorch tensors
    x_train, y_train = zip(*tfds.as_numpy(ds_train))
    x_test, y_test = zip(*tfds.as_numpy(ds_test))

    train_ds = TensorDataset(
        torch.tensor(np.asarray(x_train, dtype=np.float32)),
        torch.tensor(np.asarray(y_train, dtype=np.int64)),
    )
    test_ds = TensorDataset(
        torch.tensor(np.asarray(x_test, dtype=np.float32)),
        torch.tensor(np.asarray(y_test, dtype=np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model and optimizer
    model = HARMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with a simple evaluation each epoch
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            # The model already applies softmax, so we use log + NLLLoss
            probs = model(x_batch)
            loss = F.nll_loss(torch.log(probs + 1e-9), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # Compute accuracy on the test set
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = torch.argmax(model(x_batch), dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f}")


if __name__ == "__main__":
    main()