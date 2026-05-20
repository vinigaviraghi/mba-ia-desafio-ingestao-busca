# Desafio MBA Engenharia de Software com IA - Full Cycle


## Ingestão e Busca Semântica

Sistema RAG com LangChain, PostgreSQL (pgVector) e Google Gemini.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave da API Google (`GOOGLE_API_KEY`)

## Configuração

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edite o `.env` com sua `GOOGLE_API_KEY`. Coloque o arquivo `document.pdf` na raiz do projeto.

## Execução

1. Subir o banco de dados:

```bash
docker compose up -d
```

2. Ingerir o PDF (executar uma vez por documento):

```bash
python src/ingest.py
```

3. Iniciar o chat:

```bash
python src/chat.py
```


## Exemplo de pergunta e resposta no chat

```
Faça sua pergunta:
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: SuperTechIABrazil R$ 10.000.000,00
```

Para sair do chat, digite `sair`.


## Comandos úteis no Docker

Parar o banco (mantém os dados no volume):

```bash
docker compose stop
```

Subir novamente:

```bash
docker compose start
```

Parar e remover os containers (os dados do Postgres permanecem no volume):

```bash
docker compose down
```

Subir do zero após `down`:

```bash
docker compose up -d
```

Reiniciar em um comando (útil se a conexão falhar):

```bash
docker compose restart
```

Ver se o Postgres está rodando:

```bash
docker compose ps
```

