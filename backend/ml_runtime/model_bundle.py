"""
Model bundle management.
Loads and stores trained model artifacts from artifact directory.
"""
import json
import numpy as np
from pathlib import Path
from backend.utils.logging import logger


class ModelBundle:
    def __init__(self):
        self.gnn_model = None
        self.mlp_model = None
        self.profile_encoder = None
        self.food_embeddings = None
        self.id_maps = None
        self.shap_background = None
        self.feature_schema = None
        self.manifest = None
        self.is_loaded = False

    def load(self, artifact_dir: Path) -> bool:
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(f"No manifest found at {artifact_dir}")
            return False

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        loaded = []

        food_emb_path = artifact_dir / "food_embeddings.npy"
        if food_emb_path.exists():
            self.food_embeddings = np.load(food_emb_path)
            loaded.append(f"food_embeddings {self.food_embeddings.shape}")

        id_maps_path = artifact_dir / "id_maps.json"
        if id_maps_path.exists():
            with open(id_maps_path, "r") as f:
                self.id_maps = json.load(f)
            loaded.append(f"id_maps ({len(self.id_maps.get('food_id_map', {}))} foods)")

        try:
            mlp_path = artifact_dir / "mlp_model.pt"
            if mlp_path.exists():
                import torch
                from ml.models.suitability_mlp import SuitabilityMLP
                from ml.features.feature_schema import get_feature_names
                n_features = len(get_feature_names())
                self.mlp_model = SuitabilityMLP(
                    input_dim=n_features,
                    hidden_dims=[128, 64, 32],
                    dropout=0.3,
                )
                self.mlp_model.load_state_dict(
                    torch.load(mlp_path, map_location="cpu", weights_only=True)
                )
                self.mlp_model.eval()
                loaded.append("mlp_model")
        except Exception as e:
            logger.warning(f"Failed to load MLP model: {e}")

        try:
            pe_path = artifact_dir / "profile_encoder.pt"
            if pe_path.exists():
                import torch
                from ml.models.profile_encoder import ProfileEncoder
                self.profile_encoder = ProfileEncoder(
                    input_dim=9, hidden_dim=128, embedding_dim=64,
                )
                self.profile_encoder.load_state_dict(
                    torch.load(pe_path, map_location="cpu", weights_only=True)
                )
                self.profile_encoder.eval()
                loaded.append("profile_encoder")
        except Exception as e:
            logger.warning(f"Failed to load ProfileEncoder: {e}")

        shap_bg_path = artifact_dir / "shap_background.npy"
        if shap_bg_path.exists():
            self.shap_background = np.load(shap_bg_path)
            loaded.append(f"shap_background ({self.shap_background.shape[0]} samples)")

        feature_schema_path = artifact_dir / "feature_schema.json"
        if feature_schema_path.exists():
            with open(feature_schema_path, "r") as f:
                self.feature_schema = json.load(f)
            loaded.append("feature_schema")

        self.is_loaded = True
        logger.info(f"Loaded artifacts from {artifact_dir.name}: {', '.join(loaded)}")
        return True

    @property
    def version(self) -> str:
        if self.manifest:
            return self.manifest.get("run_id", "unknown")
        return "no_artifact"

    @property
    def food_id_map(self) -> dict:
        if self.id_maps:
            return self.id_maps.get("food_id_map", {})
        return {}

    @property
    def food_id_to_index(self) -> dict:
        return {fid: idx for fid, idx in self.food_id_map.items()}
