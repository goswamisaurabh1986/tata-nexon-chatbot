from fastapi.testclient import TestClient

from src.config.settings import Settings


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config):
        self.calls.append({"state": state, "config": config})
        return {
            "generation": "Tata Nexon has strong safety features.",
            "response": {
                "answer": "Tata Nexon has strong safety features.",
                "sources": ["brochure.pdf:1"],
                "confidence": 0.88,
                "is_grounded": True,
                "route": "final",
            },
            "citations": [{"citation_id": "brochure.pdf:1"}],
            "confidence": 0.88,
            "is_grounded": True,
            "route": "final",
            "reasoning_steps": ["Retrieved relevant safety context."],
        }


class FakeIngestionProcessor:
    def __init__(self):
        self.calls = []

    def process(self, **kwargs):
        self.calls.append(kwargs)
        metadata = {"source": "brochure.pdf", "page_count": 2}
        metadata.update(kwargs.get("metadata_overrides") or {})
        return [
            {
                "text": "Safety context",
                "metadata": metadata,
            }
        ]


class FakeBytesMetadataIngestionProcessor(FakeIngestionProcessor):
    def process(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "text": "Safety context",
                "metadata": {
                    "source": "brochure.pdf",
                    "raw_pdf_bytes": b"%PDF-1.4 bytes",
                },
            }
        ]


class FakeStreamingGraph(FakeGraph):
    async def astream_events(self, state, config, version="v2"):
        self.calls.append({"state": state, "config": config, "version": version})
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": {"content": "Safety"}},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": {"content": " features"}},
        }
        yield {
            "event": "on_chain_end",
            "data": {
                "output": {
                    "generation": "Safety features",
                    "response": {
                        "answer": "Safety features",
                        "sources": ["brochure.pdf:1"],
                        "confidence": 0.9,
                        "is_grounded": True,
                        "route": "final",
                    },
                    "reasoning_steps": ["Streamed graph events."],
                }
            },
        }


class FakeDuplicateCollection:
    def __init__(self, records=None):
        self.records = records or []
        self.get_calls = []
        self.delete_calls = []

    def get(self, where=None, include=None):
        self.get_calls.append({"where": where, "include": include})
        where = where or {}
        matches = self.records
        if "document_hash" in where:
            matches = [
                record
                for record in matches
                if record["metadata"].get("document_hash") == where["document_hash"]
            ]
        if "source" in where:
            matches = [
                record
                for record in matches
                if record["metadata"].get("source") == where["source"]
            ]

        return {
            "ids": [record["id"] for record in matches],
            "metadatas": [record["metadata"] for record in matches],
        }

    def delete(self, ids=None, where=None):
        self.delete_calls.append({"ids": ids or [], "where": where})


class FakeDuplicateStorer:
    def __init__(self, collection):
        self.collection = collection


class FakeDuplicateAwareIngestionProcessor(FakeIngestionProcessor):
    def __init__(self, collection):
        super().__init__()
        self.storer = FakeDuplicateStorer(collection)


def test_create_app_health_endpoint():
    from src.api.main import create_app

    app = create_app(settings=Settings(), graph=FakeGraph())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "Tata Nexon Chatbot"


def test_chat_endpoint_invokes_graph_with_thread_id():
    from src.api.main import create_app

    graph = FakeGraph()
    app = create_app(settings=Settings(), graph=graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "What are the Tata Nexon safety features?",
                "thread_id": "thread-123",
                "include_reasoning": True,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["thread_id"] == "thread-123"
    assert body["answer"] == "Tata Nexon has strong safety features."
    assert body["sources"] == ["brochure.pdf:1"]
    assert body["reasoning_steps"] == ["Retrieved relevant safety context."]
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == "thread-123"
    assert graph.calls[0]["state"]["query"] == "What are the Tata Nexon safety features?"


def test_chat_endpoint_generates_thread_id_when_missing():
    from src.api.main import create_app

    app = create_app(settings=Settings(), graph=FakeGraph())

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "What is the Tata Nexon mileage?"},
        )

    assert response.status_code == 200
    assert response.json()["thread_id"]


def test_chat_endpoint_streams_sse_when_requested():
    from src.api.main import create_app

    graph = FakeStreamingGraph()
    app = create_app(settings=Settings(), graph=graph)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/chat?stream=true",
            json={
                "message": "Stream Tata Nexon safety features.",
                "thread_id": "stream-thread",
                "include_reasoning": True,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "start", "thread_id": "stream-thread"}' in body
    assert '"type": "token"' in body
    assert '"content": "Safety"' in body
    assert '"type": "final"' in body
    assert '"answer": "Safety features"' in body
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == "stream-thread"


def test_chat_get_endpoint_streams_for_eventsource_clients():
    from src.api.main import create_app

    graph = FakeStreamingGraph()
    app = create_app(settings=Settings(), graph=graph)

    with TestClient(app) as client:
        with client.stream(
            "GET",
            "/chat?stream=true&message=Stream%20Tata%20Nexon%20safety&thread_id=browser-thread",
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "start", "thread_id": "browser-thread"}' in body
    assert '"type": "final"' in body
    assert '"answer": "Safety features"' in body
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == "browser-thread"


def test_admin_ingest_calls_ingestion_processor():
    from src.api.main import create_app

    processor = FakeIngestionProcessor()
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            json={
                "file_path": "brochure.pdf",
                "force_reprocess": True,
                "metadata_overrides": {"source": "brochure.pdf"},
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["chunks_created"] == 1
    assert body["chunks_stored"] == 1
    assert processor.calls[0]["file_path"] == "brochure.pdf"
    assert processor.calls[0]["chunk_size"] == 1000


def test_admin_ingest_accepts_multipart_upload(tmp_path, monkeypatch):
    from src.api.main import create_app
    from src.api.routes import admin as admin_routes

    monkeypatch.setattr(admin_routes, "ADMIN_UPLOAD_DIR", tmp_path)
    processor = FakeIngestionProcessor()
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            data={
                "force_reprocess": "true",
                "collection_name": "test_collection",
            },
            files={
                "file": (
                    "uploaded-brochure.txt",
                    b"Tata Nexon safety features.",
                    "text/plain",
                )
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["source"] == "uploaded-brochure.txt"
    assert body["metadata"]["uploaded_via"] == "frontend"
    assert body["metadata"]["force_reprocess"] is True
    assert body["metadata"]["collection_name"] == "test_collection"
    assert processor.calls[0]["source_filename"] == "uploaded-brochure.txt"
    assert tmp_path.joinpath("uploaded-brochure.txt").exists()


def test_admin_ingest_accepts_pdf_multipart_upload(tmp_path, monkeypatch):
    from src.api.main import create_app
    from src.api.routes import admin as admin_routes

    monkeypatch.setattr(admin_routes, "ADMIN_UPLOAD_DIR", tmp_path)
    processor = FakeIngestionProcessor()
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            data={"force_reprocess": "false"},
            files={
                "file": (
                    "nexon-test.pdf",
                    b"%PDF-1.4\n% test pdf bytes\n",
                    "application/pdf",
                )
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["source"] == "nexon-test.pdf"
    assert processor.calls[0]["source_filename"] == "nexon-test.pdf"
    assert processor.calls[0]["file_path"].endswith("nexon-test.pdf")
    assert tmp_path.joinpath("nexon-test.pdf").exists()


def test_admin_ingest_skips_duplicate_document_hash_without_force(tmp_path, monkeypatch):
    import hashlib

    from src.api.main import create_app
    from src.api.routes import admin as admin_routes

    file_content = b"Tata Nexon duplicate brochure."
    document_hash = hashlib.sha256(file_content).hexdigest()
    collection = FakeDuplicateCollection(
        records=[
            {
                "id": "chunk-existing-1",
                "metadata": {
                    "source": "nexon-duplicate.pdf",
                    "document_hash": document_hash,
                },
            }
        ]
    )
    processor = FakeDuplicateAwareIngestionProcessor(collection)

    monkeypatch.setattr(admin_routes, "ADMIN_UPLOAD_DIR", tmp_path)
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            data={"force_reprocess": "false"},
            files={"file": ("nexon-duplicate.pdf", file_content, "application/pdf")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "already_exists"
    assert body["chunks_created"] == 0
    assert body["chunks_stored"] == 0
    assert body["metadata"]["document_hash"] == document_hash
    assert body["metadata"]["existing_chunks"] == 1
    assert body["metadata"]["message"].startswith("Document already exists")
    assert processor.calls == []


def test_admin_ingest_passes_document_hash_to_ingestion_metadata(tmp_path, monkeypatch):
    import hashlib

    from src.api.main import create_app
    from src.api.routes import admin as admin_routes

    file_content = b"Tata Nexon new brochure."
    document_hash = hashlib.sha256(file_content).hexdigest()
    collection = FakeDuplicateCollection()
    processor = FakeDuplicateAwareIngestionProcessor(collection)

    monkeypatch.setattr(admin_routes, "ADMIN_UPLOAD_DIR", tmp_path)
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            data={"force_reprocess": "false"},
            files={"file": ("nexon-new.pdf", file_content, "application/pdf")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["metadata"]["document_hash"] == document_hash
    assert processor.calls[0]["metadata_overrides"]["document_hash"] == document_hash


def test_admin_ingest_force_reprocess_deletes_existing_source_chunks(tmp_path, monkeypatch):
    import hashlib

    from src.api.main import create_app
    from src.api.routes import admin as admin_routes

    file_content = b"Tata Nexon force reprocess brochure."
    document_hash = hashlib.sha256(file_content).hexdigest()
    collection = FakeDuplicateCollection(
        records=[
            {
                "id": "chunk-existing-1",
                "metadata": {
                    "source": "nexon-force.pdf",
                    "document_hash": document_hash,
                },
            },
            {
                "id": "chunk-existing-2",
                "metadata": {
                    "source": "nexon-force.pdf",
                    "document_hash": document_hash,
                },
            },
        ]
    )
    processor = FakeDuplicateAwareIngestionProcessor(collection)

    monkeypatch.setattr(admin_routes, "ADMIN_UPLOAD_DIR", tmp_path)
    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=processor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            data={"force_reprocess": "true"},
            files={"file": ("nexon-force.pdf", file_content, "application/pdf")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "reprocessed"
    assert body["metadata"]["document_hash"] == document_hash
    assert body["metadata"]["deleted_chunks"] == 2
    assert collection.delete_calls == [{"ids": ["chunk-existing-1", "chunk-existing-2"], "where": None}]
    assert len(processor.calls) == 1


def test_validation_error_handler_safely_serializes_bytes():
    import json

    from src.api.main import _error_response

    response = _error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details={"errors": [{"input": b"\xff", "ctx": {"raw": bytearray(b"abc")}}]},
    )

    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"]["errors"], list)
    assert body["error"]["details"]["errors"][0]["input"] == "<binary data omitted: 1 bytes>"
    assert body["error"]["details"]["errors"][0]["ctx"]["raw"] == (
        "<binary data omitted: 3 bytes>"
    )


def test_admin_ingest_response_metadata_safely_serializes_bytes():
    from src.api.main import create_app

    app = create_app(
        settings=Settings(),
        graph=FakeGraph(),
        ingestion_processor=FakeBytesMetadataIngestionProcessor(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/admin/ingest",
            json={"file_path": "brochure.pdf"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["raw_pdf_bytes"] == "<binary metadata omitted>"
