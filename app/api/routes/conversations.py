from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_conversation_repository
from app.repositories.conversation_repository import ConversationRepository
from app.schemas import AnalyticsSummary, ConversationMessage, ConversationSession

router = APIRouter()


@router.get("/sessions", response_model=list[ConversationSession])
def sessions(
    repository: Annotated[
        ConversationRepository,
        Depends(get_conversation_repository),
    ],
) -> list[ConversationSession]:
    return repository.list_sessions()


@router.get("/sessions/{session_id}/messages", response_model=list[ConversationMessage])
def session_messages(
    session_id: str,
    repository: Annotated[
        ConversationRepository,
        Depends(get_conversation_repository),
    ],
) -> list[ConversationMessage]:
    if repository.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return repository.get_messages(session_id)


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(
    repository: Annotated[
        ConversationRepository,
        Depends(get_conversation_repository),
    ],
) -> AnalyticsSummary:
    return repository.get_analytics_summary()
