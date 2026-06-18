"""
Export all trained artifacts into a versioned bundle.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def export_bundle(run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    bundle_dir = ARTIFACTS_DIR / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "components": [],
        "files": {},
    }

    gnn_dir = ARTIFACTS_DIR / "run_gnn_v1"
    if gnn_dir.exists():
        for f in gnn_dir.iterdir():
            if f.is_file() and f.name != "manifest.json":
                dest = bundle_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                manifest["components"].append("gnn")
                manifest["files"][f.name] = str(f.stat().st_size)

    mlp_dir = ARTIFACTS_DIR / "run_mlp_v1"
    if mlp_dir.exists():
        for f in mlp_dir.iterdir():
            if f.is_file() and f.name != "manifest.json":
                dest = bundle_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                if "mlp" not in manifest["components"]:
                    manifest["components"].append("mlp")
                manifest["files"][f.name] = str(f.stat().st_size)

    manifest["components"] = list(set(manifest["components"]))

    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported bundle to {bundle_dir}")
    print(f"Components: {manifest['components']}")
    print(f"Files: {list(manifest['files'].keys())}")

    return bundle_dir


if __name__ == "__main__":
    export_bundle()
