"""
Pydantic response models for the PaperQA API.

Keeping these in a separate module lets every layer (routes, tests, clients)
import the canonical shape without pulling in FastAPI or the pipeline logic.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskResponse(BaseModel):
    """Response shape for POST /ask and POST /."""

    answer: str = Field(
        default="",
        description="Extracted answer span from the document.",
    )
    context: str = Field(
        default="",
        description="Retrieved context passage that was passed to the QA model.",
    )
    char_start: int = Field(
        default=-1,
        description="Start character offset of the answer within *context* (-1 if not found).",
    )
    char_end: int = Field(
        default=-1,
        description="End character offset of the answer within *context* (-1 if not found).",
    )
    confidence: float = Field(
        default=0.0,
        description="QA model span confidence in [0, 1].",
    )
    found: bool = Field(
        default=False,
        description="True when a confident answer span was extracted.",
    )
    retrieval_confidence: float = Field(
        default=0.0,
        description=(
            "Top retrieved window's TF-IDF score normalised to [0, 1]. "
            "0.0 when no passage cleared the relevance threshold."
        ),
    )
    page_number: Optional[int] = Field(
        default=None,
        description=(
            "0-indexed PDF page number where the answer begins. "
            "null when page resolution fails or is unavailable."
        ),
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message, present only on failure responses.",
    )


class HealthResponse(BaseModel):
    status: str
    cached_docs: int
