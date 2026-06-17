"""Root & API-root info endpoints."""
from fastapi import APIRouter

from ..constants import VERSION

router = APIRouter(tags=["meta"])


@router.get("/")
async def root():
    return {"message": "ScAI - Single Cell Sequencing Analysis Driven by AI", "version": VERSION, "docs": "/docs"}


@router.get("/api")
async def api_root():
    return {"message": "ScAI API", "version": VERSION}
