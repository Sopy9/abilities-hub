# EyePop SDK — Claude Code Skill

`SKILL.md` is a **Claude Code skill** that gives Claude deep knowledge of the EyePop Python SDK.

Add it to your project and Claude will understand the full EyePop workflow — auth, dataset management, VLM ability registration, `set_pop`, and inference — without needing you to explain it.

---

## What you get

When the skill is active, Claude can:

- Write correct `dataEndpoint` / `async_worker` code from a plain description
- Register custom VLM abilities with the right parameters the first time
- Build `set_pop` pipelines including crop-then-classify (`CropForward`)
- Handle auth, retry logic, and result-reading correctly
- Reference the full pretrained ability list by name

---

## Installation

### Option 1 — Copy into your project (recommended)

```bash
mkdir -p .claude/skills
cp path/to/abilities_hub/claude/SKILL.md .claude/skills/eyepop-sdk.md
```

Claude Code automatically loads all `.md` files under `.claude/skills/` in the project root.

### Option 2 — Install globally for all projects

```bash
mkdir -p ~/.claude/skills
cp path/to/abilities_hub/claude/SKILL.md ~/.claude/skills/eyepop-sdk.md
```

### Option 3 — Clone and symlink (keeps it up to date)

```bash
git clone https://github.com/eyepop-ai/abilities_hub.git ~/eyepop-abilities
mkdir -p .claude/skills
ln -s ~/eyepop-abilities/claude/SKILL.md .claude/skills/eyepop-sdk.md
```

---

## Verify it loaded

Start a Claude Code session in your project and ask:

> "What EyePop skills do you have loaded?"

Claude will confirm the `customer-sdk` skill is active and summarize what it covers.

---

## Requirements

- [Claude Code](https://claude.ai/code) — CLI, desktop app, or IDE extension
- An EyePop API key (`eyp_...`) from [dashboard.eyepop.ai](https://dashboard.eyepop.ai)
- Python 3.12+, `eyepop` SDK installed (`pip install eyepop`)

---

## Related resources

- [EyePop Developer Docs](https://docs.eyepop.ai/developer-documentation)
- [Pretrained models and abilities](https://docs.eyepop.ai/developer-documentation/sdks/pretrained-models-and-abilities)
- [Abilities Hub](https://www.eyepop.ai/abilities) — ready-to-use published abilities
- [EyePop Dashboard](https://dashboard.eyepop.ai/dashboard) — build and train custom abilities
