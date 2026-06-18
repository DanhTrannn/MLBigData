"""
Build bipartite user-food graph from interactions.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
USERS_PATH = PROJECT_ROOT / "data" / "processed" / "users.json"
FOODS_PATH = PROJECT_ROOT / "data" / "processed" / "foods.json"
INTERACTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def build_graph():
    with open(USERS_PATH, "r") as f:
        users = json.load(f)
    with open(FOODS_PATH, "r") as f:
        foods = json.load(f)
    with open(INTERACTIONS_PATH, "r") as f:
        interactions = json.load(f)

    user_id_map = {u["user_id"]: i for i, u in enumerate(users)}
    food_id_map = {f["food_id"]: i for i, f in enumerate(foods)}

    positive_events = {"like", "eaten", "rating"}
    edges = []
    for interaction in interactions:
        if interaction["event_type"] in positive_events:
            if interaction["event_type"] == "rating" and (interaction.get("event_value") or 0) < 3:
                continue
            uid = user_id_map.get(interaction["user_id"])
            fid = food_id_map.get(interaction["food_id"])
            if uid is not None and fid is not None:
                edges.append([uid, fid])

    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge[0], edge[1])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    graph_data = {
        "num_users": len(users),
        "num_foods": len(foods),
        "num_edges": len(unique_edges),
        "edges": unique_edges,
        "user_id_map": user_id_map,
        "food_id_map": food_id_map,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "user_food_graph.json"
    with open(output_path, "w") as f:
        json.dump(graph_data, f)

    print(f"Built graph: {len(users)} users, {len(foods)} foods, {len(unique_edges)} edges")
    return graph_data


if __name__ == "__main__":
    build_graph()
