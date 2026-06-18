"""Public chat endpoint.

The chat route is the API-facing adapter around the LangGraph agent. It accepts
validated HTTP input, chooses or creates a memory ``thread_id``, invokes the
compiled graph, and normalizes graph state into a stable ``ChatResponse``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from src.agent.memory import generate_thread_id, graph_config
from src.agent.schemas import AgentResponse
from src.agent.state import AgentState, initial_agent_state
from src.api.dependencies import get_agent_graph, get_settings
from src.api.schemas import ChatRequest, ChatResponse
from src.config.settings import Settings


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    stream: bool = Query(False, description="Stream the response as Server-Sent Events."),
    graph: Any = Depends(get_agent_graph),
    settings: Settings = Depends(get_settings),
) -> ChatResponse | StreamingResponse:
    """Run the LangGraph agent for one user message.

    Args:
        request: Validated chat request body.
        graph: Compiled LangGraph application dependency.
        settings: Typed application settings dependency.

    Returns:
        ChatResponse containing the answer, citations, thread ID, confidence,
        route, and optional reasoning.

    Raises:
        HTTPException: Returned as a safe 503 response when graph invocation
            fails.
    """
    thread_id = _thread_id_for_request(request, settings)
    state = _agent_state_for_request(request, settings)

    if stream:
        return _streaming_response(
            graph=graph,
            state=state,
            thread_id=thread_id,
            include_reasoning=request.include_reasoning,
        )

    result = await _invoke_graph(graph, state, thread_id)

    return _chat_response_from_state(
        result,
        thread_id=thread_id,
        include_reasoning=request.include_reasoning,
    )


@router.get("/chat", response_class=StreamingResponse)
def chat_event_stream(
    message: str = Query(..., min_length=1, description="User message to stream."),
    stream: bool = Query(True, description="Must be true for EventSource clients."),
    thread_id: str | None = Query(None, description="Conversation thread ID."),
    user_id: str | None = Query(None, description="User/session owner."),
    top_k: int | None = Query(None, ge=1, le=50),
    include_reasoning: bool = Query(False),
    graph: Any = Depends(get_agent_graph),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream chat responses from a browser-native ``EventSource`` request.

    Browsers only allow ``EventSource`` to issue GET requests, so this endpoint
    mirrors ``POST /chat?stream=true`` with query parameters while keeping
    non-streaming chat on the POST endpoint.
    """
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use POST /chat for non-streaming chat.",
        )

    request = ChatRequest(
        message=message,
        thread_id=thread_id,
        user_id=user_id,
        top_k=top_k,
        include_reasoning=include_reasoning,
    )
    active_thread_id = _thread_id_for_request(request, settings)
    state = _agent_state_for_request(request, settings)

    return _streaming_response(
        graph=graph,
        state=state,
        thread_id=active_thread_id,
        include_reasoning=include_reasoning,
    )


def _streaming_response(
    graph: Any,
    state: AgentState,
    thread_id: str,
    include_reasoning: bool,
) -> StreamingResponse:
    """Build a reusable SSE response for POST and EventSource clients."""
    return StreamingResponse(
        _stream_chat_response(
            graph=graph,
            state=state,
            thread_id=thread_id,
            include_reasoning=include_reasoning,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _invoke_graph(graph: Any, state: AgentState, thread_id: str) -> dict[str, Any]:
    """Invoke the graph and translate failures into HTTP errors.

    Async checkpointers require LangGraph's async runtime methods. Tests and
    simple fakes may still expose only ``invoke``, so this helper supports both.
    """
    try:
        if hasattr(graph, "ainvoke"):
            return await graph.ainvoke(state, config=graph_config(thread_id))
        return graph.invoke(state, config=graph_config(thread_id))
    except Exception as error:
        logger.exception("Chat graph invocation failed for thread %s.", thread_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat agent is temporarily unavailable.",
        ) from error


async def _stream_chat_response(
    graph: Any,
    state: AgentState,
    thread_id: str,
    include_reasoning: bool,
) -> AsyncIterator[str]:
    """Stream graph execution as Server-Sent Events.

    The stream prefers LangGraph's ``astream_events`` API so token-level model
    events can be forwarded when the underlying LLM supports streaming. If the
    graph only exposes ``astream`` or ``invoke``, the function still emits a
    valid SSE stream with a final response event.
    """
    yield _sse_event({"type": "start", "thread_id": thread_id})

    final_state: dict[str, Any] | None = None
    try:
        if hasattr(graph, "astream_events"):
            async for event in _iter_astream_events(graph, state, thread_id):
                token = _token_from_event(event)
                if token:
                    yield _sse_event(
                        {
                            "type": "token",
                            "thread_id": thread_id,
                            "content": token,
                        }
                    )

                candidate_state = _final_state_from_event(event)
                if candidate_state is not None:
                    final_state = candidate_state
        elif hasattr(graph, "astream"):
            final_state = await _stream_from_astream(graph, state, thread_id)
        else:
            final_state = graph.invoke(state, config=graph_config(thread_id))

        if final_state is not None:
            final_response = _chat_response_from_state(
                final_state,
                thread_id=thread_id,
                include_reasoning=include_reasoning,
            )
            yield _sse_event({"type": "final", **final_response.model_dump()})

        yield _sse_event({"type": "done", "thread_id": thread_id})
    except Exception:
        logger.exception("Streaming chat graph invocation failed for thread %s.", thread_id)
        yield _sse_event(
            {
                "type": "error",
                "thread_id": thread_id,
                "message": "Chat agent is temporarily unavailable.",
            }
        )


async def _iter_astream_events(
    graph: Any,
    state: AgentState,
    thread_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield LangGraph event dictionaries using the newest supported API."""
    try:
        event_stream = graph.astream_events(
            state,
            config=graph_config(thread_id),
            version="v2",
        )
    except TypeError:
        event_stream = graph.astream_events(state, config=graph_config(thread_id))

    async for event in event_stream:
        yield event


async def _stream_from_astream(
    graph: Any,
    state: AgentState,
    thread_id: str,
) -> dict[str, Any] | None:
    """Stream LangGraph state updates and return the latest state-like output."""
    final_state: dict[str, Any] | None = None
    async for update in graph.astream(state, config=graph_config(thread_id)):
        candidate_state = _state_like_mapping(update)
        if candidate_state is not None:
            final_state = candidate_state
    return final_state


def _token_from_event(event: dict[str, Any]) -> str | None:
    """Extract token text from a LangGraph/LangChain stream event."""
    data = event.get("data", {})
    chunk = data.get("chunk") if isinstance(data, dict) else None
    if chunk is None and isinstance(data, dict):
        chunk = data.get("message")
    return _content_from_chunk(chunk)


def _content_from_chunk(chunk: Any) -> str | None:
    """Extract text content from common LangChain chunk shapes."""
    if chunk is None:
        return None
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return _content_to_text(chunk.get("content") or chunk.get("text"))
    return _content_to_text(
        getattr(chunk, "content", None) or getattr(chunk, "text", None)
    )


def _content_to_text(content: Any) -> str | None:
    """Convert chunk content into text, including list-based content blocks."""
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        text = "".join(parts)
        return text or None
    return str(content) if content else None


def _final_state_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Return state-like output from a terminal LangGraph event."""
    data = event.get("data", {})
    if not isinstance(data, dict):
        return None
    return _state_like_mapping(data.get("output"))


def _state_like_mapping(value: Any) -> dict[str, Any] | None:
    """Find a graph-state-shaped mapping inside nested stream updates."""
    if not isinstance(value, dict):
        return None

    state_keys = {
        "generation",
        "response",
        "citations",
        "confidence",
        "route",
        "is_grounded",
        "reasoning_steps",
    }
    if any(key in value for key in state_keys):
        return value

    for nested_value in value.values():
        nested_state = _state_like_mapping(nested_value)
        if nested_state is not None:
            return nested_state
    return None


def _sse_event(payload: dict[str, Any]) -> str:
    """Format a payload as one Server-Sent Event data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _thread_id_for_request(request: ChatRequest, settings: Settings) -> str:
    """Return the caller-provided thread ID or create a new one."""
    if request.thread_id:
        return request.thread_id
    return generate_thread_id(request.user_id or settings.memory.default_user_id)


def _agent_state_for_request(request: ChatRequest, settings: Settings) -> AgentState:
    """Build the initial LangGraph state for a chat request."""
    state = initial_agent_state(
        request.message,
        messages=[("human", request.message)],
    )
    state["top_k"] = request.top_k or settings.retrieval.top_k
    return state


def _chat_response_from_state(
    result: dict[str, Any],
    thread_id: str,
    include_reasoning: bool,
) -> ChatResponse:
    """Normalize LangGraph output into the public API response model."""
    response = result.get("response")
    reasoning_steps = _reasoning_steps(result, include_reasoning)

    if isinstance(response, AgentResponse):
        return _response_from_agent_response(response, result, thread_id, reasoning_steps)
    if isinstance(response, dict):
        return _response_from_mapping(response, result, thread_id, reasoning_steps)
    return _response_from_fallback(result, thread_id, reasoning_steps)


def _response_from_agent_response(
    response: AgentResponse,
    result: dict[str, Any],
    thread_id: str,
    reasoning_steps: list[str],
) -> ChatResponse:
    """Build a ChatResponse from the canonical AgentResponse model."""
    return ChatResponse(
        answer=response.answer,
        thread_id=thread_id,
        sources=response.sources,
        confidence=response.confidence,
        is_grounded=response.is_grounded,
        route=response.route or result.get("route"),
        reasoning_steps=reasoning_steps,
    )


def _response_from_mapping(
    response: dict[str, Any],
    result: dict[str, Any],
    thread_id: str,
    reasoning_steps: list[str],
) -> ChatResponse:
    """Build a ChatResponse from dict-like graph response state."""
    return ChatResponse(
        answer=str(response.get("answer") or result.get("generation") or ""),
        thread_id=thread_id,
        sources=[str(source) for source in response.get("sources", [])],
        confidence=_safe_float(response.get("confidence", result.get("confidence", 0.0))),
        is_grounded=bool(response.get("is_grounded", result.get("is_grounded", False))),
        route=response.get("route") or result.get("route"),
        reasoning_steps=reasoning_steps,
    )


def _response_from_fallback(
    result: dict[str, Any],
    thread_id: str,
    reasoning_steps: list[str],
) -> ChatResponse:
    """Build a ChatResponse when the graph produced only primitive fields."""
    response = result.get("response")
    return ChatResponse(
        answer=str(response or result.get("generation") or "No response was generated."),
        thread_id=thread_id,
        sources=_sources_from_state(result),
        confidence=_safe_float(result.get("confidence", 0.0)),
        is_grounded=bool(result.get("is_grounded", False)),
        route=result.get("route"),
        reasoning_steps=reasoning_steps,
    )


def _sources_from_state(result: dict[str, Any]) -> list[str]:
    """Extract citation IDs from graph state."""
    sources: list[str] = []
    for citation in result.get("citations", []):
        if isinstance(citation, dict):
            source = citation.get("citation_id") or citation.get("source")
            if source:
                sources.append(str(source))
        else:
            sources.append(str(citation))
    return sources


def _reasoning_steps(result: dict[str, Any], include_reasoning: bool) -> list[str]:
    """Return reasoning steps only when explicitly requested."""
    if not include_reasoning:
        return []
    return [str(step) for step in result.get("reasoning_steps", [])]


def _safe_float(value: Any) -> float:
    """Convert a numeric-like value to float with a stable fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
