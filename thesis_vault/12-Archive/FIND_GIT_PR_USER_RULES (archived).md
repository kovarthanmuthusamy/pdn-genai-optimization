---
title: FIND_GIT_PR_USER_RULES (archived)
type: archive
source: docs/_archive/FIND_GIT_PR_USER_RULES.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Where the git/PR rules live (and how to make them on-demand)

## Found

Your agent still receives two **always-on User Rules** every turn:

| Internal name | Est. size | When it should load |
|---------------|-----------|---------------------|
| `committing-changes-with-git` | ~600–700 tokens | Only when you ask to commit |
| `creating-pull-requests` | ~400–600 tokens | Only when you ask to open a PR |

They appear in the agent context as:

```xml
<user_rule><committing-changes-with-git>…</user_rule>
<user_rule><creating-pull-requests>…</user_rule>
```

## Where they are stored

| Location | Result |
|----------|--------|
| `genai_pdn/.cursor/` | **Not here** — no git/PR rules |
| `~/.cursor/skills/git-commit/` etc. | On-demand skills only (`disable-model-invocation: true`) — **not** the source of the always-on text |
| WSL `~/.config/Cursor/` | **Empty** — Cursor runs on Windows, not Linux |
| Windows `C:\Users\muthusamy\AppData\Roaming\Cursor\User\globalStorage\state.vscdb` | Chat caches reference the rules; **canonical copy is account-synced User Rules** (Cursor cloud), not a plain file you can edit |

**Why Settings looked empty:** these are **named User Rules** in the **Customize → Rules** UI (newer Cursor), not a single “Rules for AI” text box. They may not show under old **Settings → Rules** paths.

## Remove always-on rules (if they appear in UI)

**Update:** If **Customize → Rules** is empty, you likely **cannot delete these in the UI** — see “If Customize is empty” below.

1. In Cursor, open **Customize** in the left sidebar.
2. Go to **Rules** → check **User**, **Project**, and **Team** tabs separately.
3. Also try **Cursor Settings** (gear) → search **Rules** (some versions still have a User Rules text field).
4. If logged into a team, check **cursor.com** dashboard → Team Rules.

If you find rules named `committing-changes-with-git` or `creating-pull-requests`, delete them and reload.

## If Customize is empty (your case)

The git/PR text is **still injected every agent turn**, but we found **no local copy** to edit:

| Checked | Result |
|---------|--------|
| `Customize → Rules` | Empty (per you) |
| `settings.json` | No rules |
| `state.vscdb` reactive storage | No rules payload |
| Repo / WSL `~/.cursor/` | Only on-demand skills + project `token-efficiency.mdc` |

**Conclusion:** Those blocks are almost certainly **Cursor account / platform rules** (synced from Cursor’s backend when Agent starts), not files in your project. The names `committing-changes-with-git` and `creating-pull-requests` match Cursor’s standard agent git/PR tooling templates.

You did **not** misconfigure anything — there may simply be **nothing to delete** in Customize.

### What you can still do

1. **Cursor support / forum** — ask to disable always-on `committing-changes-with-git` and `creating-pull-requests` for your account (context bloat, not visible in Customize).
2. **Disable browser MCP** (Settings → Tools & MCP) — saves more tokens than git rules (~16 tools).
3. **New chat per task** — avoids history bloat.
4. **On-demand skills** (already on disk) — use when you ask to commit or open a PR; they do not remove the duplicate always-on text until Cursor stops injecting it.

## On-demand replacement (already on disk)

After deleting the User Rules, use these only when needed:

| Task | Skill |
|------|-------|
| Git commit | `~/.cursor/skills/git-commit/SKILL.md` |
| Pull request | `~/.cursor/skills/creating-pull-requests/SKILL.md` |

Say e.g. “commit these changes” or “open a PR” — the agent should load the skill then.

Optional one-line User Rule after cleanup:

```
Git commits: when asked, follow ~/.cursor/skills/git-commit/SKILL.md. PRs: when asked, follow ~/.cursor/skills/creating-pull-requests/SKILL.md.
```

## Why the agent could not auto-delete them

The `manage_personal_rules` / `cursor_dialog` MCP tools (list/remove user rules) are **not enabled in this session**. Rules are also synced from your Cursor account, so editing WSL files would not remove them.

## Verify after removal

Start a **new agent chat** and ask: “List my user rules related to git or PR.”  
If removal worked, those two large blocks should be gone (~900–1,300 tokens saved per turn).
