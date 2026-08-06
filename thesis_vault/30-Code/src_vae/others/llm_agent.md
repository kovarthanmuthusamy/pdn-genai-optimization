---
title: llm_agent
type: code
path: src_vae/others/llm_agent.py
group: src_vae/others
loc: 173
tags: [code, src_vae]
---

# llm_agent

> DeepSeek LLM agent for autonomous training analysis and config fixes.

**Source:** `src_vae/others/llm_agent.py` · 173 lines

## Purpose

```text
DeepSeek LLM agent for autonomous training analysis and config fixes.

Run: ``python -m src_vae.others.llm_agent`` (smoke test) or import ``analyze_training_issues``.
```

## Constants

| Name | Value |
|------|-------|
| `DEEPSEEK_BASE_URL` | `'https://api.deepseek.com/v1'` |
| `DEEPSEEK_MODEL` | `'deepseek-chat'` |
| `SYSTEM_PROMPT` | `'You are an expert ML training diagnostician for a multi-modal VAE doing PDN (Power Deliv…` |

## Functions

- **`_load_api_key()`**
- **`call_deepseek(messages, model=DEEPSEEK_MODEL, temperature=0.3, max_tokens=2048, api_key=None, timeout=60)`**
- **`analyze_training_issues(issues, actions_taken, snapshot, history_summary, config_summary, api_key=None)`**
- **`apply_llm_suggestions(config_changes, exp_dir, dry_run=False)`**

## External dependencies

`requests`
