# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Local HTTP transport tests, without a model service or retrieval-quality assertions."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from openviking.models.embedder import OpenAIDenseEmbedder
from openviking.models.embedder.base import embed_compat
from openviking.utils.async_client_cache import LoopScopedAsyncClientCache
from openviking_cli.utils.config.embedding_config import EmbeddingConfig, EmbeddingModelConfig


@pytest.fixture
def embeddings_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            response = json.dumps(
                {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "model": "test-embedding"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("provider", ["openai", "azure", "ollama"])
@pytest.mark.parametrize("credentials", [[], [{"id": "primary", "api_key": "local-key"}]])
@pytest.mark.parametrize("method", ["embed", "embed_async"])
async def test_query_instruction_on_wire(embeddings_endpoint, provider, credentials, method):
    api_base, requests = embeddings_endpoint
    prefix = "Instruct: Match passages\nQuery: "
    config = EmbeddingModelConfig(
        provider=provider,
        model="test-embedding",
        api_key="local-key",
        api_base=api_base,
        dimension=2,
        query_instruction=prefix,
        credentials=credentials,
    )
    embedder = EmbeddingConfig(dense=config).get_embedder()
    batch = ["first", "second"]
    try:
        for content, is_query, expected in [
            ("first", True, prefix + "first"),
            ("first", False, "first"),
            (batch, True, [prefix + "first", prefix + "second"]),
            (batch, False, batch),
        ]:
            result = getattr(embedder, method)(content, is_query=is_query)
            if method == "embed_async":
                result = await result
            assert result.dense_vector == [0.1, 0.2]
            path, body = requests[-1]
            assert "embeddings" in path
            assert body["input"] == expected
            assert body["model"] == "test-embedding"
        assert batch == ["first", "second"]
        assert len(requests) == 4
    finally:
        await LoopScopedAsyncClientCache.close_current_loop_clients()
        embedder.close()


@pytest.mark.parametrize("method", ["embed", "embed_async"])
@pytest.mark.parametrize("prefix", [None, "", "Q: "])
async def test_query_instruction_input_scope_and_budget(embeddings_endpoint, method, prefix):
    api_base, requests = embeddings_endpoint
    config = EmbeddingModelConfig(
        provider="openai", model="test-embedding", api_base=api_base, dimension=2
    )
    assert config.query_instruction is None
    config.query_instruction = prefix
    text_embedder = EmbeddingConfig(dense=config, max_input_tokens=1).get_embedder()
    embedder = OpenAIDenseEmbedder(
        model_name="test-embedding",
        api_base=api_base,
        dimension=2,
        input_type="multimodal",
        query_instruction=prefix,
        config={"max_input_tokens": 1},
    )
    parts = [
        {"type": "text", "text": "xy"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    expected = "Q: x\n...(truncated for embedding)" if prefix else "xy"
    try:
        for content, wire_input in [
            ("xy", expected),
            (["xy", "xy"], [expected, expected]),
            (parts, parts),
        ]:
            result = getattr(embedder, method)(content, is_query=True)
            if method == "embed_async":
                await result
            assert requests[-1][1]["input"] == wire_input
        result = getattr(text_embedder, method)([{"type": "text", "text": "xy"}], is_query=True)
        if method == "embed_async":
            await result
        assert requests[-1][1]["input"] == "xy"
        await embed_compat(text_embedder, [{"type": "text", "text": "xy"}], is_query=True)
        assert requests[-1][1]["input"] == expected
    finally:
        await LoopScopedAsyncClientCache.close_current_loop_clients()
        embedder.close()
        text_embedder.close()
