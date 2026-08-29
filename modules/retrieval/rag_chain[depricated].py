import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


try:
    from modules.pinecone.pinecone_store import VectorStore
    from modules.nvidia.nvidia_llm import get_llm
except ModuleNotFoundError:
    try:
        from modules.retrieval.pinecone_store import VectorStore
        from modules.nvidia.nvidia_llm import get_llm
    except ModuleNotFoundError:
        try:
            from modules.retrieval.vector_store import VectorStore
            from modules.llm.language_model import get_llm
        except ModuleNotFoundError:
            from vector_store import VectorStore
            from language_model import get_llm


def format_docs(docs):
    """TODO: join retrieved chunks into context string"""
    # hint: "\n\n".join(d.page_content for d in docs)
    # optional: include source e.g. f"[{d.metadata.get('source')}] {d.page_content}"
    return "\n\n".join(doc.page_content for doc in docs)


PROMPT = ChatPromptTemplate.from_template(
    """You are Sift, answering only from the provided paper context.

Context:
{context}

Question: {question}

Rules:
- Answer strictly from context, no outside knowledge.
- If not in context, say "Not found in the paper."
- Cite source file names where possible.

Answer:"""
)


def get_retrieval_chain(k: int = 4):


    # 1. vector store
    vs = VectorStore()
    retriever = vs.vector_store.as_retriever(search_kwargs={"k": k})

    # 2. llm
    llm = get_llm()

    # 3. chain (LCEL)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain