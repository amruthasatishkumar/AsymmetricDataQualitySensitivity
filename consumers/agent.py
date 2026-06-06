"""LLM agent harness for AgentDQ.

Wraps AzureOpenAI (Entra ID auth) with:
- profiled access mode (schema + per-column training-set summary)
- strict JSON output validation
- retry-on-malformed-output
- JSONL logging of every call to logs/agent/

The agent never sees row labels at inference time.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from consumers.features import T1_EXTRA_FEATURE_COLS, T2_FEATURE_COLS, T3_TEXT_COLS
from consumers.profile import get_or_make_profile
from consumers.prompts.templates import build_prompt

REPO = Path(__file__).resolve().parent.parent
LOGS_AGENT = REPO / "logs" / "agent"

CATEGORICAL_COLS = {"customer_state", "payment_type_modal", "dominant_category", "order_status"}


def _load_env() -> None:
    """Read .env without external deps; ignore BOM."""
    p = REPO / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# Load .env at import time so AgentConfig defaults can read os.environ
_load_env()


@dataclass
class AgentConfig:
    task: str  # "t1", "t2", or "t3"
    deployment: str = field(default_factory=lambda: os.environ["AZURE_AI_FOUNDRY_DEPLOYMENT"])
    api_version: str = field(default_factory=lambda: os.environ.get("AZURE_AI_FOUNDRY_API_VERSION", "2024-08-01-preview"))
    endpoint: str = field(default_factory=lambda: os.environ["AZURE_AI_FOUNDRY_ENDPOINT"])
    temperature: float = 0.0
    max_tokens: int = 200
    seed: int = 13
    log_tag: str = "default"


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_strict(text: str) -> dict:
    """Parse {prediction, confidence, rationale} or raise ValueError."""
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fences if model ignored instructions
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise ValueError(f"no JSON object in: {text[:200]!r}")
        obj = json.loads(m.group(0))
    if "prediction" not in obj or "confidence" not in obj:
        raise ValueError(f"missing keys: {obj!r}")
    pred = int(obj["prediction"])
    conf = float(obj["confidence"])
    if pred not in (0, 1):
        raise ValueError(f"prediction must be 0 or 1: {pred!r}")
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence out of [0,1]: {conf!r}")
    return {
        "prediction": pred,
        "confidence": conf,
        "rationale": str(obj.get("rationale", ""))[:400],
    }


class Agent:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=cfg.api_version,
        )

        if cfg.task == "t1":
            self.feature_cols = list(T2_FEATURE_COLS) + list(T1_EXTRA_FEATURE_COLS)
        elif cfg.task == "t3":
            # T3 uses tabular order features + Portuguese review text
            self.feature_cols = list(T2_FEATURE_COLS) + list(T3_TEXT_COLS)
        else:
            self.feature_cols = list(T2_FEATURE_COLS)

        self._profile: str = get_or_make_profile(cfg.task, self.feature_cols, CATEGORICAL_COLS)

        LOGS_AGENT.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = LOGS_AGENT / f"{cfg.task}_a1_{cfg.log_tag}_{ts}.jsonl"

    def _build_messages(self, row: dict) -> list[dict]:
        return build_prompt(self.cfg.task, row, self.feature_cols, self._profile)

    def predict_one(self, row: dict, retries: int = 1) -> dict:
        messages = self._build_messages(row)
        last_err: str | None = None
        for attempt in range(retries + 1):
            t0 = time.time()
            # 429 retry loop with exponential backoff (caps at ~32s)
            backoff = 1.0
            resp = None
            for rl_attempt in range(8):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.cfg.deployment,
                        messages=messages,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    is_rate = ("429" in msg or "Too Many Requests" in msg
                               or "rate" in msg.lower())
                    if not is_rate or rl_attempt == 7:
                        raise
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 32.0)
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            try:
                parsed = _parse_strict(content)
                ok = True
                err = None
            except Exception as e:  # noqa: BLE001
                parsed = {"prediction": -1, "confidence": 0.0, "rationale": ""}
                ok = False
                err = str(e)
                last_err = err
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "task": self.cfg.task,
                "mode": "a1",
                "attempt": attempt,
                "ok": ok,
                "error": err,
                "latency_s": round(time.time() - t0, 3),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "raw": content,
                "parsed": parsed,
            }
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if ok:
                return parsed
        return {"prediction": -1, "confidence": 0.0, "rationale": f"PARSE_FAIL: {last_err}"}

    def predict_batch(self, rows: Iterable[dict]) -> list[dict]:
        out = []
        for i, row in enumerate(rows):
            r = self.predict_one(row)
            out.append(r)
        return out


TEXT_FIELD_MAX_CHARS = 800  # cap T3 review text per field to bound prompt size


def _row_dict(s: pd.Series, feature_cols: list[str]) -> dict:
    d = {}
    for c in feature_cols:
        v = s.get(c)
        if pd.isna(v):
            d[c] = None
        elif hasattr(v, "item"):
            d[c] = v.item()
        else:
            d[c] = v
        # Truncate free-text fields to keep prompts bounded
        if isinstance(d[c], str) and len(d[c]) > TEXT_FIELD_MAX_CHARS:
            d[c] = d[c][:TEXT_FIELD_MAX_CHARS] + "…"
    return d


def main() -> int:
    """Pilot: run the A1 agent on a stratified sample."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2", "t3"], default="t3")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", type=str, default="pilot")
    args = ap.parse_args()

    cfg = AgentConfig(task=args.task, log_tag=args.tag, seed=args.seed)
    agent = Agent(cfg)

    df = pd.read_parquet(REPO / "data" / "silver" / "order_features.parquet")
    target = {"t1": "y_t1", "t2": "y_t2", "t3": "y_t3"}[args.task]
    pool = df[(df["split"] == args.split) & df[target].notna()]
    half = args.n // 2
    pos = pool[pool[target] == 1].sample(n=half, random_state=args.seed)
    neg = pool[pool[target] == 0].sample(n=args.n - half, random_state=args.seed)
    sub = pd.concat([pos, neg]).sample(frac=1.0, random_state=args.seed)

    n_ok = 0
    correct = 0
    total_tokens = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for _, row in sub.iterrows():
        d = _row_dict(row, agent.feature_cols)
        r = agent.predict_one(d)
        if r["prediction"] in (0, 1):
            n_ok += 1
            y_true.append(int(row[target]))
            y_pred.append(r["prediction"])
            if r["prediction"] == int(row[target]):
                correct += 1
    # quick accuracy from log
    for line in agent.log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        total_tokens += rec.get("total_tokens", 0)

    # per-class breakdown
    pos_total = sum(1 for y in y_true if y == 1)
    neg_total = sum(1 for y in y_true if y == 0)
    pos_correct = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    neg_correct = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    pred_pos = sum(1 for p in y_pred if p == 1)

    print(f"\n[{args.task}/a1] {args.n} rows from {args.split} (stratified 50/50)")
    print(f"  parsed_ok       : {n_ok}/{args.n}")
    print(f"  accuracy        : {correct}/{n_ok}  "
          f"({(correct / n_ok) if n_ok else 0:.2%})")
    print(f"  TPR (pos recall): {pos_correct}/{pos_total}  "
          f"({(pos_correct / pos_total) if pos_total else 0:.2%})")
    print(f"  TNR (neg recall): {neg_correct}/{neg_total}  "
          f"({(neg_correct / neg_total) if neg_total else 0:.2%})")
    print(f"  predicted-positive rate: {pred_pos}/{n_ok}")
    print(f"  total_tokens    : {total_tokens:,}")
    print(f"  log             : {agent.log_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
