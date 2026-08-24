from dotenv import load_dotenv

load_dotenv()

try:
    from modules.retrieval.rag_chain import get_retrieval_chain
except ModuleNotFoundError:
    try:
        from modules.retrieval.retrieval_chain import get_retrieval_chain
    except ModuleNotFoundError:
        from modules.retrieval_chain import get_retrieval_chain


chain = get_retrieval_chain(k=4)

answer = chain.invoke("What is attention is all you need?")
print(answer)
print("---")

# hallucination guard test — not in paper
print(chain.invoke("Who wrote Hamilton musical?"))