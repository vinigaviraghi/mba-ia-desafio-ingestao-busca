import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LLM_MODEL = "gemini-2.5-flash-lite"
SEARCH_K = 10

_vector_store = None
_llm = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "mba_docs"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )
    return _vector_store


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return _llm


def build_context(results) -> str:
    return "\n\n".join(doc.page_content for doc, _ in results)


def _run_search(question: str) -> str:
    store = get_vector_store()
    results = store.similarity_search_with_score(question, k=SEARCH_K)
    contexto = build_context(results)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
    response = get_llm().invoke(prompt)
    return response.content.strip()


def search_prompt(question: str | None = None) -> Callable[[str], str] | str | None:
    try:
        get_vector_store()
        get_llm()
    except Exception as exc:
        print(f"Erro na inicialização: {exc}")
        return None

    if question is not None:
        return _run_search(question)

    return _run_search
