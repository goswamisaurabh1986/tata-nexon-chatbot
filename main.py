"""User-friendly interactive CLI for the Tata Nexon LangGraph chatbot."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agent.graph import build_agent_graph
from src.agent.memory import (
    generate_thread_id,
    get_or_create_thread_id,
    graph_config,
    list_thread_ids,
    save_thread_id,
)
from src.agent.schemas import AgentResponse
from src.agent.state import initial_agent_state
from src.config.settings import Settings, load_settings
from src.ingestion.embedder import Embedder
from src.ingestion.storer import VectorStorer
from src.retrieval.retriever import Retriever


DEFAULT_CHAT_MODEL = "gpt-4o-mini"


try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


class OpenAICompatibleAgentResponse(BaseModel):
    """OpenAI structured-output compatible version of AgentResponse."""

    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    is_grounded: bool
    reasoning_steps: list[str]
    refusal_reason: str | None
    route: str | None


class OpenAIStructuredLLM:
    """Tiny adapter that gives OpenAI the LangChain structured-output shape."""

    def __init__(self, api_key: str, model: str = DEFAULT_CHAT_MODEL) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def with_structured_output(self, schema: type[BaseModel]):
        """Return an invoker that parses responses into the requested schema."""
        response_schema = OpenAICompatibleAgentResponse if schema is AgentResponse else schema
        return OpenAIStructuredInvoker(
            client=self.client,
            model=self.model,
            target_schema=schema,
            response_schema=response_schema,
        )


class OpenAIStructuredInvoker:
    """Invoke OpenAI chat completions with Pydantic structured output."""

    def __init__(
        self,
        client: Any,
        model: str,
        target_schema: type[BaseModel],
        response_schema: type[BaseModel],
    ) -> None:
        self.client = client
        self.model = model
        self.target_schema = target_schema
        self.response_schema = response_schema

    def invoke(self, messages: list[tuple[str, str]]) -> BaseModel:
        """Call the model and coerce parsed output into target_schema."""
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": _openai_role(role), "content": content}
                for role, content in messages
            ],
            response_format=self.response_schema,
            temperature=0,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return self.target_schema.model_validate(message.parsed.model_dump())
        return self.target_schema.model_validate_json(message.content)

    def batch(self, messages_batch: list[list[tuple[str, str]]]) -> list[BaseModel]:
        """Support nodes that prefer batch structured-output calls."""
        return [self.invoke(messages) for messages in messages_batch]


@dataclass
class ChatSession:
    """Runtime session state for the interactive CLI."""

    user_id: str
    thread_id: str
    show_reasoning: bool = False


def main() -> None:
    """Start the interactive chatbot."""
    _load_environment()
    settings = load_settings()
    _configure_logging(settings)

    llm = _build_llm(settings)
    retriever = _build_retriever(settings)
    graph, checkpointer = build_agent_graph(
        llm=llm,
        retriever=retriever,
        settings=settings,
        return_checkpointer=True,
    )

    if graph is None:
        _print_error("LangGraph is not installed. Run: pip install langgraph")
        return

    session = _load_chat_session(settings)
    _print_welcome(session, llm=llm, retriever=retriever, checkpointer=checkpointer)
    _chat_loop(
        graph=graph,
        llm=llm,
        retriever=retriever,
        session=session,
        settings=settings,
    )


def _load_environment() -> None:
    """Load environment variables from project env files."""
    load_dotenv()
    load_dotenv(".env.txt", override=False)


def _configure_logging(settings: Settings) -> None:
    """Configure quiet logging for interactive use."""
    logging.basicConfig(level=getattr(logging, settings.app.log_level, logging.WARNING))


def _build_llm(settings: Settings) -> OpenAIStructuredLLM | None:
    """Create the structured-output LLM when OPENAI_API_KEY is available."""
    api_key = settings.llm.api_key
    if not api_key:
        return None

    try:
        return OpenAIStructuredLLM(
            api_key=api_key,
            model=settings.llm.chat_model,
        )
    except Exception as error:
        _print_warning(f"Could not initialize OpenAI chat model: {error}")
        return None


def _build_retriever(settings: Settings) -> Retriever:
    """Create the real retriever, falling back to demo retrieval if needed."""
    if not settings.llm.api_key:
        return Retriever()

    try:
        return Retriever(
            embedder=Embedder(
                model_name=settings.llm.embedding_model,
                api_key=settings.llm.api_key,
                dimensions=settings.llm.embedding_dimensions,
            ),
            vector_store=VectorStorer(
                collection_name=settings.retrieval.collection_name,
                persist_directory=settings.retrieval.persist_directory,
                embedding_dimension=settings.retrieval.embedding_dimension,
            ),
        )
    except Exception as error:
        _print_warning(f"Could not initialize Chroma retriever; using demo fallback: {error}")
        return Retriever()


def _load_chat_session(settings: Settings) -> ChatSession:
    """Resume the previous thread for the configured user."""
    user_id = settings.memory.default_user_id
    thread_id = get_or_create_thread_id(user_id)
    return ChatSession(user_id=user_id, thread_id=thread_id)


def _chat_loop(
    graph: Any,
    llm: Any | None,
    retriever: Retriever,
    session: ChatSession,
    settings: Settings,
) -> None:
    """Run the interactive read/evaluate/print loop."""
    while True:
        try:
            query = input(f"\n[{session.thread_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print_goodbye()
            return

        if not query:
            continue
        if _handle_command(query, session):
            continue

        state = initial_agent_state(query, messages=[("human", query)])
        state["llm"] = llm
        state["retriever"] = retriever
        state["top_k"] = settings.retrieval.top_k

        try:
            result = graph.invoke(state, config=graph_config(session.thread_id))
        except Exception as error:
            _print_error(f"Agent error: {error}")
            continue

        _print_result(result, show_reasoning=session.show_reasoning)


def _handle_command(command: str, session: ChatSession) -> bool:
    """Handle CLI commands. Return true when command was consumed."""
    normalized = command.strip()
    lower_command = normalized.lower()

    if lower_command in {"exit", "quit", "q"}:
        _print_goodbye()
        raise SystemExit(0)
    if lower_command in {"help", "/help", "?"}:
        _print_help()
        return True
    if lower_command in {"new", "/new"}:
        session.thread_id = generate_thread_id(session.user_id)
        save_thread_id(session.user_id, session.thread_id)
        _print_success(f"Started new conversation: {session.thread_id}")
        return True
    if lower_command in {"threads", "/threads", "chats", "/chats"}:
        _print_threads(session)
        return True
    if lower_command in {"thread", "/thread", "session", "/session"}:
        _print_info(f"Current conversation: {session.thread_id}")
        return True
    if lower_command.startswith(("continue ", "/continue ", "thread ", "/thread ")):
        session.thread_id = normalized.split(maxsplit=1)[1].strip()
        save_thread_id(session.user_id, session.thread_id)
        _print_success(f"Continued conversation: {session.thread_id}")
        return True
    if lower_command in {"/reason", "reason"}:
        session.show_reasoning = not session.show_reasoning
        status = "on" if session.show_reasoning else "off"
        _print_info(f"Reasoning display is now {status}.")
        return True
    if lower_command in {"/reason on", "reason on"}:
        session.show_reasoning = True
        _print_info("Reasoning display is now on.")
        return True
    if lower_command in {"/reason off", "reason off"}:
        session.show_reasoning = False
        _print_info("Reasoning display is now off.")
        return True

    return False


def _print_welcome(
    session: ChatSession,
    llm: Any | None,
    retriever: Retriever,
    checkpointer: Any | None,
) -> None:
    """Print a friendly startup screen."""
    retrieval_mode = "ChromaDB" if retriever.vector_store is not None else "demo fallback"
    lines = [
        "Ask Tata Nexon brochure/product questions.",
        "",
        f"User: {session.user_id}",
        f"Conversation: {session.thread_id}",
        f"LLM: {'OpenAI structured output' if llm else 'not configured'}",
        f"Retrieval: {retrieval_mode}",
        f"Memory: {'checkpointer enabled' if checkpointer else 'disabled'}",
        "",
        "Commands: new, continue <thread_id>, threads, /reason, help, quit",
    ]
    _print_panel("Tata Nexon Chatbot", "\n".join(lines))


def _print_help() -> None:
    """Print CLI help."""
    lines = [
        "new                  Start a fresh conversation",
        "continue <thread_id> Continue a previous conversation",
        "threads              Show known conversations for this user",
        "/reason              Toggle reasoning steps in answers",
        "help                 Show this help message",
        "exit or quit         Close the chatbot",
        "",
        "Example: What are the safety features of Tata Nexon?",
    ]
    _print_panel("Commands", "\n".join(lines))


def _print_threads(session: ChatSession) -> None:
    """Print known conversation IDs for the current user."""
    thread_ids = list_thread_ids(session.user_id)
    if not thread_ids:
        _print_info("No saved conversations found.")
        return

    if RICH_AVAILABLE:
        table = Table(title="Conversations")
        table.add_column("Active")
        table.add_column("Thread ID")
        for thread_id in thread_ids:
            table.add_row("*" if thread_id == session.thread_id else "", thread_id)
        console.print(table)
        return

    print("\nConversations")
    print("-" * 32)
    for thread_id in thread_ids:
        marker = "*" if thread_id == session.thread_id else " "
        print(f"{marker} {thread_id}")


def _print_result(result: dict[str, Any], show_reasoning: bool) -> None:
    """Print a formatted chatbot response."""
    answer = _answer_from_result(result)
    sources = _sources_from_result(result)
    confidence = _confidence_from_result(result)
    reasoning_steps = result.get("reasoning_steps", [])

    body = answer
    if confidence is not None:
        body += f"\n\nConfidence: {confidence:.2f}"
    if sources:
        body += "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
    if show_reasoning and reasoning_steps:
        body += "\n\nReasoning:\n" + "\n".join(
            f"- {step}" for step in reasoning_steps[-8:]
        )

    _print_panel("Answer", body)


def _answer_from_result(result: dict[str, Any]) -> str:
    """Extract answer text from graph state."""
    response = result.get("response")
    if isinstance(response, AgentResponse):
        return response.answer
    if isinstance(response, dict):
        return str(response.get("answer") or result.get("generation") or "")
    if isinstance(response, str) and response.strip():
        return response
    return str(result.get("generation") or "No response was generated.")


def _sources_from_result(result: dict[str, Any]) -> list[str]:
    """Extract source/citation IDs from graph state."""
    response = result.get("response")
    if isinstance(response, AgentResponse) and response.sources:
        return response.sources
    if isinstance(response, dict) and response.get("sources"):
        return [str(source) for source in response["sources"]]

    sources = []
    for citation in result.get("citations", []):
        if isinstance(citation, dict):
            source = citation.get("citation_id") or citation.get("source")
            if source:
                sources.append(str(source))
        else:
            sources.append(str(citation))
    return sources


def _confidence_from_result(result: dict[str, Any]) -> float | None:
    """Extract response confidence from graph state."""
    response = result.get("response")
    if isinstance(response, AgentResponse):
        return response.confidence
    if isinstance(response, dict) and "confidence" in response:
        return float(response["confidence"])
    if "confidence" in result:
        return float(result["confidence"])
    return None


def _print_panel(title: str, body: str) -> None:
    """Print a panel with rich when available, otherwise plain text."""
    if RICH_AVAILABLE:
        console.print(Panel(body, title=title, border_style="cyan"))
        return

    print(f"\n{title}")
    print("=" * len(title))
    print(body)


def _print_success(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[green]{message}[/green]")
    else:
        print(message)


def _print_info(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def _print_warning(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[yellow]{message}[/yellow]")
    else:
        print(f"Warning: {message}")


def _print_error(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[red]{message}[/red]")
    else:
        print(f"Error: {message}")


def _print_goodbye() -> None:
    _print_info("Goodbye.")


def _openai_role(role: str) -> str:
    """Map LangChain-style roles to OpenAI chat roles."""
    if role in {"human", "user"}:
        return "user"
    if role in {"system", "assistant", "developer"}:
        return role
    return "user"


if __name__ == "__main__":
    main()
