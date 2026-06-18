"""
Train MLP suitability model.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "mlp_features.npy"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "mlp_labels.npy"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def train():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)["mlp"]

    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    torch.manual_seed(42)

    n = len(X)
    indices = np.random.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1)

    from ml.models.suitability_mlp import SuitabilityMLP
    model = SuitabilityMLP(
        input_dim=X.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.MSELoss()

    batch_size = config["batch_size"]
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_X = X_train[perm[start:end]]
            batch_y = y_train[perm[start:end]]

            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()
        mae = (test_pred - y_test).abs().mean().item()

    print(f"\nTest MSE: {test_loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    run_id = "run_mlp_v1"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "mlp_model.pt")

    from ml.features.feature_schema import get_schema_info
    schema_info = get_schema_info()
    with open(run_dir / "feature_schema.json", "w") as f:
        json.dump(schema_info, f, indent=2)

    shap_bg_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
    np.save(run_dir / "shap_background.npy", X[shap_bg_idx])

    manifest = {
        "run_id": run_id,
        "model_type": "suitability_mlp",
        "input_dim": X.shape[1],
        "hidden_dims": config["hidden_dims"],
        "test_mse": test_loss,
        "test_mae": mae,
        "num_samples": n,
    }

    existing_manifest = run_dir / "manifest.json"
    if existing_manifest.exists():
        with open(existing_manifest, "r") as f:
            existing = json.load(f)
        existing.update(manifest)
        manifest = existing

    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    train()
