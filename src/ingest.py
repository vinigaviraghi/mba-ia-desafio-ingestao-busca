import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_embeddings

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Evita estourar TPM do Gemini (30k/min no tier gratuito)
EMBED_BATCH_SIZE = 10
EMBED_BATCH_DELAY_SEC = 10


def resolve_path(path: str | None) -> str | None:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def ingest_pdf():
    pdf_path = resolve_path(os.getenv("PDF_PATH"))
    if not pdf_path or not os.path.isfile(pdf_path):
        print(f"PDF não encontrado: {pdf_path}")
        sys.exit(1)

    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    total = len(chunks)

    if total == 0:
        print("Nenhum trecho gerado a partir do PDF.")
        sys.exit(1)

    embedding = get_embeddings()
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    connection = os.getenv("DATABASE_URL")
    store = None

    total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    print(
        "Na primeira execução pode aparecer 'Collection not found' no log — "
        "é normal (a collection ainda não existia naquele momento)."
    )

    if total > EMBED_BATCH_SIZE:
        print(
            f"Processando {total} trechos em {total_batches} lotes de {EMBED_BATCH_SIZE} "
            f"(pausa de {EMBED_BATCH_DELAY_SEC}s entre lotes)..."
        )

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        end = min(start + EMBED_BATCH_SIZE, total)
        batch_num = start // EMBED_BATCH_SIZE + 1
        print(f"Lote {batch_num}/{total_batches}: salvando trechos {start + 1}-{end} de {total}...")

        if store is None:
            store = PGVector.from_documents(
                documents=batch,
                embedding=embedding,
                collection_name=collection_name,
                connection=connection,
                use_jsonb=True,
                pre_delete_collection=True,
            )
        else:
            store.add_documents(batch)

        print(f"Lote {batch_num}/{total_batches} concluído.")

        if end < total:
            time.sleep(EMBED_BATCH_DELAY_SEC)

    print(f"PDF processado com sucesso: {total} trechos salvos no banco de dados.")


if __name__ == "__main__":
    ingest_pdf()
