import os
from langchain_openai import ChatOpenAI


def get_llm(
    nvidia_api_key: str | None = None,
    model: str = "nvidia/nemotron-3-ultra-550b-a55b",
    temperature: float = 0.2,
):
    """
    Returns ChatOpenAI pointed at NVIDIA.
    Step: resolve api_key (arg > env), return ChatOpenAI(...)
    """
    nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY missing")

    llm = ChatOpenAI(
        api_key=nvidia_api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        model=model,
        temperature=temperature,
    )
    return llm
