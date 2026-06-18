"""
Artifact loader.
Loads model bundle at startup and provides readiness check.
"""
from pathlib import Path
from backend.ml_runtime.model_bundle import ModelBundle
from backend.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


class ArtifactLoader:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.bundle = ModelBundle()

    def load_latest(self) -> ModelBundle:
        if not self.artifacts_dir.exists():
            logger.info("No artifacts directory found, running in baseline mode")
            return self.bundle

        run_dirs = sorted(
            [d for d in self.artifacts_dir.iterdir() if d.is_dir()],
            key=lambda d: len(list(d.iterdir())),
            reverse=True,
        )

        if not run_dirs:
            logger.info("No artifact runs found, running in baseline mode")
            return self.bundle

        best = run_dirs[0]
        logger.info(f"Loading best artifact: {best.name} ({len(list(best.iterdir()))} files)")
        self.bundle.load(best)

        for run_dir in run_dirs[1:]:
            if not run_dir.is_dir():
                continue
            self._merge_missing(run_dir)

        return self.bundle

    def _merge_missing(self, run_dir: Path):
        import numpy as np
        import json

        if self.bundle.food_embeddings is None:
            emb_path = run_dir / "food_embeddings.npy"
            if emb_path.exists():
                self.bundle.food_embeddings = np.load(emb_path)
                logger.info(f"Merged food_embeddings from {run_dir.name}")

        if self.bundle.id_maps is None:
            maps_path = run_dir / "id_maps.json"
            if maps_path.exists():
                with open(maps_path, "r") as f:
                    self.bundle.id_maps = json.load(f)
                logger.info(f"Merged id_maps from {run_dir.name}")

    def is_ready(self) -> bool:
        return True
