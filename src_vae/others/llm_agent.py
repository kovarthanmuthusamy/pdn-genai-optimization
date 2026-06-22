"""DeepSeek LLM agent for autonomous training analysis and config fixes.

Run: ``python -m src_vae.others.llm_agent`` (smoke test) or import ``analyze_training_issues``."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


def _load_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def call_deepseek(messages, model=DEEPSEEK_MODEL, temperature=0.3,
                  max_tokens=2048, api_key=None, timeout=60):
    key = api_key or _load_api_key()
    if not key:
        print("  [LLM] ERROR: DEEPSEEK_API_KEY not set")
        return None
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"  [LLM] API error {resp.status_code}: {resp.text[:300]}")
            return None
        return resp.json()
    except requests.exceptions.Timeout:
        print("  [LLM] API timeout")
        return None
    except Exception as e:
        print(f"  [LLM] API call failed: {e}")
        return None


SYSTEM_PROMPT = """You are an expert ML training diagnostician for a multi-modal VAE doing PDN (Power Delivery Network) design.

Modalities: Heatmap (64x64x2, Huber+grad+Laplacian), Occupancy (52 binary slots, focal BCE+margin), Impedance (231-point log-impedance, Huber+concavity+topK).

You can suggest changes to: learning_rate, lr_patience, lr_min, occupancy_weight, impedance_weight, heatmap_weight, beta_final, beta_phase2_final, free_bits, sigma_reg_weight, sigma_reg_target, occupancy_focal_gamma, focal_gamma_warmup_frac, num_epochs.

Respond with concise analysis and CONFIG: suggestions (one per line)."""


def analyze_training_issues(issues, actions_taken, snapshot, history_summary,
                            config_summary, api_key=None):
    hist_lines = []
    for key, vals in history_summary.items():
        if isinstance(vals, list) and len(vals) >= 2:
            hist_lines.append(f"  {key}: {vals[0]:.3f} -> {vals[1]:.3f} ({vals[2]} pts)")
        elif isinstance(vals, list) and len(vals) == 1:
            hist_lines.append(f"  {key}: {vals[0]:.3f}")

    snap = "\n".join(f"  {k}: {v}" for k, v in snapshot.items()
                     if isinstance(v, (int, float)))

    cfg = "\n".join(f"  {k}: {v}" for k, v in config_summary.items())

    prompt = f"""## Issues
{chr(10).join(f'- {i}' for i in issues)}

## Actions Taken
{chr(10).join(f'- {a}' for a in actions_taken)}

## Snapshot
{snap}

## History
{chr(10).join(hist_lines) if hist_lines else 'N/A'}

## Config
{cfg}

Analyze and suggest concrete CONFIG: changes (one per line, format: CONFIG: key: value)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    print("  [LLM] Calling DeepSeek API...")
    response = call_deepseek(messages, api_key=api_key)

    if response is None:
        return {"analysis": "API call failed", "config_changes": {},
                "model_changes_needed": False, "raw_response": None}

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  [LLM] Response: {len(content)} chars")

    config_changes = {}
    for line in content.splitlines():
        m = re.match(r"^\s*CONFIG:\s*(\w+):\s*(.+?)\s*$", line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            try:
                if "." in val or "e" in val.lower():
                    config_changes[key] = f"{float(val):.4f}"
                else:
                    config_changes[key] = str(int(val))
            except ValueError:
                config_changes[key] = val

    return {"analysis": content, "config_changes": config_changes,
            "model_changes_needed": False, "raw_response": response}


def apply_llm_suggestions(config_changes, exp_dir, dry_run=False):
    config_path = Path(exp_dir) / "config.yaml"
    if not config_path.exists():
        return ["config.yaml not found"]

    content = config_path.read_text()
    applied = []

    for key, new_val in config_changes.items():
        pattern = rf"^(\s*{re.escape(key)}:\s*)(\S.*)$"
        match = re.search(pattern, content, flags=re.MULTILINE)
        if match:
            old_val = match.group(2).strip()
            if old_val != new_val:
                if dry_run:
                    applied.append(f"  [DRY RUN] {key}: {old_val} -> {new_val}")
                else:
                    content = re.sub(pattern, rf"\g<1>{new_val}", content, flags=re.MULTILINE)
                    applied.append(f"  [LLM] {key}: {old_val} -> {new_val}")
        else:
            applied.append(f"  [SKIP] {key}: not found in config.yaml")

    if not dry_run and any("->" in a for a in applied):
        config_path.write_text(content)

    return applied


if __name__ == "__main__":
    key = _load_api_key()
    if not key:
        print("DEEPSEEK_API_KEY not set. Create .env file with: DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    result = analyze_training_issues(
        issues=["FLAT: occupancy_loss at 0.228 (0.3% over 40 epochs)",
                "LR_FLOOR: lr=5.00e-06 <= warning 2.00e-05"],
        actions_taken=["AUTO: Raised lr_min"],
        snapshot={"total_loss": 2.72, "occupancy_loss": 0.228, "kl_loss": 0.90},
        history_summary={"val_occupancy_loss": [0.60, 0.228, 20]},
        config_summary={"num_epochs": 600, "learning_rate": 1e-4,
                        "occupancy_weight": 5.0, "beta_phase2_final": 0.15},
    )
    print("\nConfig changes:", result["config_changes"])
