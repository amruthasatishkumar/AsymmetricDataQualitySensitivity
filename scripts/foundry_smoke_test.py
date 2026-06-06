"""Smoke test for the Azure AI Foundry / Azure OpenAI deployment.

Run this once after filling in `.env`:
    python scripts/foundry_smoke_test.py

It does ONE chat completion call with temperature=0 and prints the result.
If this fails, fix credentials/network before doing anything else.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if not env_path.exists():
        print(f"[FAIL] .env not found at {env_path}. Copy .env.example to .env and fill values.")
        return 2
    load_dotenv(env_path)

    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", "").strip()
    deployment = os.environ.get("AZURE_AI_FOUNDRY_DEPLOYMENT", "").strip()
    api_version = os.environ.get("AZURE_AI_FOUNDRY_API_VERSION", "2024-08-01-preview").strip()

    missing = [k for k, v in {
        "AZURE_AI_FOUNDRY_ENDPOINT": endpoint,
        "AZURE_AI_FOUNDRY_DEPLOYMENT": deployment,
    }.items() if not v]
    if missing:
        print(f"[FAIL] Missing env vars: {missing}")
        return 2

    print(f"Endpoint   : {endpoint}")
    print(f"Deployment : {deployment}")
    print(f"API version: {api_version}")
    print("Auth       : Entra ID (DefaultAzureCredential)")
    print("Calling chat.completions ...")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    resp = client.chat.completions.create(
        model=deployment,
        temperature=0,
        max_tokens=64,
        messages=[
            {"role": "system", "content": "You return strict JSON."},
            {
                "role": "user",
                "content": (
                    "Return JSON with keys 'ok' (bool, true) and 'echo' (string, value 'agentdq')."
                ),
            },
        ],
    )

    content = resp.choices[0].message.content or ""
    usage = resp.usage
    print("\n--- response.content ---")
    print(content)
    print("\n--- usage ---")
    print(f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")

    # Try to parse strict JSON
    try:
        parsed = json.loads(content)
        assert parsed.get("ok") is True
        assert parsed.get("echo") == "agentdq"
        print("\n[OK] Smoke test passed. Foundry deployment is reachable and returns parseable JSON.")
        return 0
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"\n[WARN] Reachable, but JSON contract not satisfied: {e}")
        print("       Connectivity is fine; we'll harden the prompt in consumers/agent.py.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
