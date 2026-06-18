"""
Evaluation module for GNN and MLP models.
"""
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_gnn(run_dir: Path) -> dict:
    from ml.models.lightgcn import LightGCN

    with open(run_dir / "manifest.json", "r") as f:
        manifest = json.load(f)

    with open(run_dir / "id_maps.json", "r") as f:
        id_maps = json.load(f)

    from data.processed import user_food_graph
    graph_path = PROJECT_ROOT / "data" / "processed" / "user_food_graph.json"
    with open(graph_path, "r") as f:
        graph = json.load(f)

    model = LightGCN(
        num_users=graph["num_users"],
        num_items=graph["num_foods"],
        embedding_dim=manifest["embedding_dim"],
        num_layers=manifest["num_layers"],
    )
    model.load_state_dict(torch.load(run_dir / "gnn_model.pt", weights_only=True))
    model.eval()

    embeddings = np.load(run_dir / "food_embeddings.npy")

    return {
        "model_type": "lightgcn",
        "num_users": graph["num_users"],
        "num_foods": graph["num_foods"],
        "num_edges": graph["num_edges"],
        "embedding_dim": manifest["embedding_dim"],
        "best_val_recall": manifest.get("best_val_recall", 0),
        "food_embeddings_shape": list(embeddings.shape),
    }


def evaluate_mlp(run_dir: Path) -> dict:
    from ml.models.suitability_mlp import SuitabilityMLP
    from ml.features.feature_schema import get_feature_names

    with open(run_dir / "manifest.json", "r") as f:
        manifest = json.load(f)

    features_path = PROJECT_ROOT / "data" / "processed" / "mlp_features.npy"
    labels_path = PROJECT_ROOT / "data" / "processed" / "mlp_labels.npy"

    if not features_path.exists():
        return {"error": "Test data not found"}

    X = np.load(features_path)
    y = np.load(labels_path)

    model = SuitabilityMLP(
        input_dim=X.shape[1],
        hidden_dims=manifest.get("hidden_dims", [128, 64, 32]),
    )
    model.load_state_dict(torch.load(run_dir / "mlp_model.pt", weights_only=True))
    model.eval()

    test_n = int(len(X) * 0.1)
    X_test = torch.tensor(X[-test_n:], dtype=torch.float32)
    y_test = torch.tensor(y[-test_n:], dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        pred = model(X_test)
        mse = ((pred - y_test) ** 2).mean().item()
        mae = (pred - y_test).abs().mean().item()

    return {
        "model_type": "suitability_mlp",
        "test_samples": test_n,
        "test_mse": round(mse, 4),
        "test_mae": round(mae, 4),
        "num_features": X.shape[1],
    }


def evaluate_all():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for run_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        model_type = manifest.get("model_type", "unknown")
        if model_type == "lightgcn":
            results[run_dir.name] = evaluate_gnn(run_dir)
        elif model_type == "suitability_mlp":
            results[run_dir.name] = evaluate_mlp(run_dir)

    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation results:")
    for run_id, metrics in results.items():
        print(f"  {run_id}: {metrics}")

    return results


if __name__ == "__main__":
    evaluate_all()
