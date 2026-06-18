"""
Train LightGCN model on user-food interaction graph.
"""
import json
import random
import numpy as np
import torch
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "user_food_graph.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def train():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)["gnn"]

    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)

    num_users = graph["num_users"]
    num_foods = graph["num_foods"]
    edges = graph["edges"]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    from ml.models.lightgcn import LightGCN

    model = LightGCN(
        num_users=num_users,
        num_items=num_foods,
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
    )

    edge_tensor = torch.tensor(edges, dtype=torch.long).t()

    n_train = int(len(edges) * config["train_split"])
    n_val = int(len(edges) * config["val_split"])

    indices = list(range(len(edges)))
    random.shuffle(indices)

    train_edges = edge_tensor[:, indices[:n_train]]
    val_edges = edge_tensor[:, indices[n_train:n_train + n_val]]

    train_user_ids = train_edges[0]
    train_item_ids = train_edges[1]

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    batch_size = config["batch_size"]
    best_val_recall = 0.0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(train_user_ids))
        for start in range(0, len(train_user_ids), batch_size):
            end = min(start + batch_size, len(train_user_ids))
            batch_users = train_user_ids[perm[start:end]]
            batch_pos = train_item_ids[perm[start:end]]

            neg_items = torch.randint(0, num_foods, (len(batch_users),))

            loss = model.bpr_loss(batch_users, batch_pos, neg_items, train_edges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model(train_edges)
                val_users = val_edges[0][:100]
                val_items = val_edges[1][:100]

                hits = 0
                for i in range(len(val_users)):
                    u = user_emb[val_users[i]]
                    scores = (item_emb * u).sum(dim=-1)
                    top_k = scores.topk(20).indices
                    if val_items[i] in top_k:
                        hits += 1

                recall = hits / max(len(val_users), 1)

            print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f} | Recall@20: {recall:.4f}")

            if recall > best_val_recall:
                best_val_recall = recall
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(train_edges)

    run_id = "run_gnn_v1"
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "gnn_model.pt")
    np.save(run_dir / "food_embeddings.npy", item_emb.numpy())

    id_maps = {
        "user_id_map": graph["user_id_map"],
        "food_id_map": graph["food_id_map"],
    }
    with open(run_dir / "id_maps.json", "w") as f:
        json.dump(id_maps, f)

    manifest = {
        "run_id": run_id,
        "model_type": "lightgcn",
        "num_users": num_users,
        "num_foods": num_foods,
        "num_edges": len(edges),
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "best_val_recall": best_val_recall,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nArtifacts saved to {run_dir}")
    print(f"Best Recall@20: {best_val_recall:.4f}")
    print(f"Food embeddings shape: {item_emb.shape}")


if __name__ == "__main__":
    train()
