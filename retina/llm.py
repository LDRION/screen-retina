"""Shared Ollama call helper."""
import ollama


def call_chat(client: ollama.Client, think: dict, **kw):
    """client.chat that drops the `think` flag once if the model does not support it."""
    try:
        return client.chat(**kw, **think)
    except ollama.ResponseError as e:
        if think and "think" in str(e).lower():
            think.clear()
            return client.chat(**kw)
        raise
