"""The LLM behind LLaMEA, by provider. LLaMEA itself is not modified: the classes here wrap or
subclass the ones in llamea.llm.

    anthropic  Claude through the Anthropic API (ANTHROPIC_API_KEY), as in the main study.
    ollama     A local or remote Ollama server (open-weight models). The host comes from
               --ollama-host or OLLAMA_HOST (default http://localhost:11434).

Why a subclass for Ollama: llamea.llm.Ollama_LLM calls ollama.chat() without options, so the
server's default context window applies (a few thousand tokens), and Ollama silently truncates
longer prompts -- LLaMEA's prompts (task, code of the parent, feedback) are longer than that.
This version sets num_ctx explicitly, talks to a chosen host, and keeps LLaMEA's retry logic.
"""
from __future__ import annotations

import os
import time

PROVIDERS = ("anthropic", "ollama")


def make_llm(provider: str, model: str, temperature=None, ollama_host: str | None = None,
             num_ctx: int = 32768):
    if provider == "anthropic":
        import anthropic
        from llamea.llm import Anthropic_LLM
        return Anthropic_LLM(api_key=os.environ["ANTHROPIC_API_KEY"], model=model,
                             temperature=anthropic.NOT_GIVEN if temperature is None else temperature)
    if provider == "ollama":
        return OllamaLLM(model=model, host=ollama_host, num_ctx=num_ctx, temperature=temperature)
    raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")


def _ollama_base():
    from llamea.llm import Ollama_LLM
    return Ollama_LLM


class OllamaLLM(_ollama_base()):
    """llamea.llm.Ollama_LLM with an explicit host and context window (see module docstring)."""

    def __init__(self, model, host=None, num_ctx=32768, temperature=None, **kwargs):
        super().__init__(model=model, **kwargs)
        import ollama
        self.host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.client = ollama.Client(host=self.host)
        self.options = {"num_ctx": int(num_ctx)}
        if temperature is not None:
            self.options["temperature"] = float(temperature)

    # LLaMEA pickles itself (warm start) and the ollama client is not picklable: drop it and
    # recreate it, as llamea.llm does for its OpenAI and Gemini clients
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        import ollama
        self.__dict__.update(state)
        self.client = ollama.Client(host=self.host)

    def __deepcopy__(self, memo):
        import copy
        new = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k != "client":
                setattr(new, k, copy.deepcopy(v, memo))
        import ollama
        new.client = ollama.Client(host=new.host)
        return new

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        import ollama
        # same message handling as Ollama_LLM: the conversation as one user message
        big_message = "".join(msg["content"] + "\n" for msg in session_messages)
        attempt = 0
        while True:
            try:
                response = self.client.chat(model=self.model, options=self.options,
                                            messages=[{"role": "user", "content": big_message}])
                return response["message"]["content"]
            except ollama.ResponseError as err:
                attempt += 1
                if attempt > max_retries or err.status_code not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)