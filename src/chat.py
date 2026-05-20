from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:")
    print("(digite 'sair' para encerrar)")

    try:
        while True:
            question = input("PERGUNTA: ").strip()
            if not question:
                continue

            if question.lower() in ("sair", "exit", "q"):
                break

            try:
                response = chain(question)
                print(f"RESPOSTA: {response}")
            except Exception as exc:
                print(f"RESPOSTA: Erro ao processar pergunta: {exc}")

    except KeyboardInterrupt:
        print("\nEncerrando chat.")


if __name__ == "__main__":
    main()
