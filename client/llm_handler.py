import os

from langchain_openai import ChatOpenAI


def get_nvidia_key(nvidia_api_key: str | None = None) -> str:
    """Resolve NVIDIA API key: arg > env var."""
    key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY missing")
    return key


def get_llm(
    nvidia_api_key: str | None = None,
    model: str = "nvidia/nemotron-3-ultra-550b-a55b",
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Return ChatOpenAI pointed at NVIDIA API."""
    key = get_nvidia_key(nvidia_api_key)
    return ChatOpenAI(
        api_key=key,
        base_url="https://integrate.api.nvidia.com/v1",
        model=model,
        temperature=temperature,
    )
