"""LangChain fica nas bordas: módulos não precisam conhecer um fornecedor de LLM."""

from __future__ import annotations

import hashlib
import json
import math
import os

from rag_lab.config import Settings


class Providers:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat = None
        self._embeddings = None

    @property
    def embedding_key(self) -> str:
        config = self.settings.embeddings
        identity = [config.provider, config.base_url, config.model, config.dimensions]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    @staticmethod
    def api_key(config) -> str:
        key = os.getenv(config.api_key_env)
        if not key:
            raise ValueError(f"Defina {config.api_key_env} para o endpoint compatível com OpenAI.")
        return key

    def chat(self, system: str, user: str) -> str:
        config = self.settings.llm
        if self._chat is None:
            if config.provider == "ollama":
                from langchain_ollama import ChatOllama

                self._chat = ChatOllama(
                    model=config.model,
                    base_url=config.base_url,
                    temperature=config.temperature,
                    client_kwargs={"timeout": config.timeout},
                )
            else:
                from langchain_openai import ChatOpenAI

                self._chat = ChatOpenAI(
                    model=config.model,
                    base_url=config.base_url,
                    api_key=self.api_key(config),
                    temperature=config.temperature,
                    timeout=config.timeout,
                    max_retries=1,
                )
        # .text também lida com modelos que retornam blocos de conteúdo.
        return self._chat.invoke([("system", system), ("human", user)]).text

    def embed(self, texts: list[str]) -> list[list[float]]:
        config = self.settings.embeddings
        if self._embeddings is None:
            if config.provider == "ollama":
                from langchain_ollama import OllamaEmbeddings

                self._embeddings = OllamaEmbeddings(
                    model=config.model,
                    base_url=config.base_url,
                    client_kwargs={"timeout": config.timeout},
                )
            else:
                from langchain_openai import OpenAIEmbeddings

                # Não força dimensions no request: alguns servidores compatíveis não suportam.
                self._embeddings = OpenAIEmbeddings(
                    model=config.model,
                    base_url=config.base_url,
                    api_key=self.api_key(config),
                    request_timeout=config.timeout,
                    max_retries=1,
                    check_embedding_ctx_length=False,
                )
        vectors = self._embeddings.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("O endpoint não devolveu um embedding por texto.")
        for vector in vectors:
            if len(vector) != config.dimensions or not all(math.isfinite(x) for x in vector):
                raise ValueError(
                    "Dimensão/valores de embedding inválidos. Confira o YAML e o modelo."
                )
            if not any(vector):
                raise ValueError("Embedding nulo não admite similaridade por cosseno.")
        return vectors
