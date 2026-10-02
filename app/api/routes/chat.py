import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_service
from app.rag.service import RAGService
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    try:
        response = rag_service.answer(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Controlled runtime error while answering chat request")
        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while answering chat request")
        raise HTTPException(status_code=500, detail="Unexpected internal error") from exc

    return ChatResponse(
        answer=response.answer,
        supported_by_context=response.supported_by_context,
        sources=response.sources,
        metadata=response.retrieval_metadata,
    )
