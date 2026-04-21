"""Root & API-root info endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/")
async def root():
    return {"message": "ScAI - Single Cell Sequencing Analysis Driven by AI", "version": "1.0.0", "docs": "/docs"}


@router.get("/api")
async def api_root():
    return {"message": "ScAI API", "version": "1.0.0"}
