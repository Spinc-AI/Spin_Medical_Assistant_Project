"""Minimal terminal chat loop — your first end-to-end test.

Run this AFTER the SSH tunnel to the server is open (see SERVER_SETUP.md):
    python chat.py

Type a question (Persian or English). Ctrl+C to quit.
"""
from llm_client import chat

# A system prompt sets the model's role. This is a placeholder you'll later
# enrich with retrieved RAG references and outputs from the other modules.
SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant helping physicians. "
    "You answer in the language the doctor uses (Persian or English). "
    "Be precise, cite uncertainty, and never replace the physician's judgment."
)


def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Medical LLM ready. Ask something (Ctrl+C to quit).\n")

    try:
        while True:
            user = input("doctor> ").strip()
            if not user:
                continue
            history.append({"role": "user", "content": user})

            print("assistant> ", end="", flush=True)
            reply = ""
            for piece in chat(history, stream=True):
                print(piece, end="", flush=True)
                reply += piece
            print("\n")
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
