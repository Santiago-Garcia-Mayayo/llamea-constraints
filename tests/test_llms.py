"""The Ollama provider: requests reach the chosen host with the context window set, and the LLM
survives LLaMEA's pickling and deep copies. Uses a tiny local HTTP server, no real model."""
import copy
import http.server
import json
import pickle
import threading

import pytest

pytest.importorskip("ollama")

from harness.llms import OllamaLLM, make_llm  # noqa: E402


@pytest.fixture
def fake_server():
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = json.dumps({"model": body["model"], "created_at": "2026-01-01T00:00:00Z",
                              "message": {"role": "assistant", "content": "hello"}, "done": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_ollama_query_sets_context_window(fake_server):
    host, seen = fake_server
    llm = make_llm("ollama", "some-model", ollama_host=host, num_ctx=16384)
    assert llm.query([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == "hello"
    assert seen[0]["model"] == "some-model"
    assert seen[0]["options"]["num_ctx"] == 16384
    assert seen[0]["messages"][0]["content"] == "a\nb\n"


def test_ollama_llm_pickles_and_copies(fake_server):
    host, _ = fake_server
    llm = OllamaLLM("m", host=host)
    for other in (pickle.loads(pickle.dumps(llm)), copy.deepcopy(llm)):
        assert other.host == host and other.options == llm.options
        assert other.query([{"role": "user", "content": "x"}]) == "hello"


def test_llm_timeout_extends_llamea_wait():
    from llamea.llm import Dummy_LLM
    from harness.llamea_adapter import make_llamea
    base = make_llamea("free", Dummy_LLM(), "toy/basic", budget=2, n_parents=1, n_offspring=1)
    slow = make_llamea("free", Dummy_LLM(), "toy/basic", budget=2, n_parents=1, n_offspring=1, llm_timeout=1800)
    assert slow.eval_timeout == base.eval_timeout + 1800