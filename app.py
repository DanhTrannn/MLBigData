"""
Meal Recommender - FastAPI Application.
Serves API endpoints and static frontend files.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.api import recommendations, feedback, catalog
from backend.services.orchestrator import RecommendationOrchestrator
from backend.storage.file_store import FileStore
from backend.ml_runtime.artifact_loader import ArtifactLoader
from backend.utils.logging import logger
from backend.utils.exceptions import RecommenderError

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

orchestrator: RecommendationOrchestrator | None = None
file_store: FileStore | None = None
artifact_loader: ArtifactLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, file_store, artifact_loader

    logger.info("Starting Meal Recommender...")

    file_store = FileStore()
    logger.info("File store initialized")

    artifact_loader = ArtifactLoader()
    bundle = artifact_loader.load_latest()
    logger.info(f"Artifact loader ready: version={bundle.version}")

    orchestrator = RecommendationOrchestrator(bundle=bundle)
    orchestrator.set_file_store(file_store)
    logger.info("Orchestrator initialized")

    logger.info("Meal Recommender started successfully!")
    yield
    logger.info("Shutting down Meal Recommender")


app = FastAPI(
    title="Meal Recommender API",
    description="Hệ thống gợi ý thực đơn 1 ngày / Daily Meal Recommendation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations.router)
app.include_router(feedback.router)
app.include_router(catalog.router)


@app.exception_handler(RecommenderError)
async def recommender_error_handler(request: Request, exc: RecommenderError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    is_ready = artifact_loader.is_ready() if artifact_loader else False
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_loaded": orchestrator is not None,
        "artifacts_ready": is_ready,
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
