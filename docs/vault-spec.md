# House of Knowledge — Obsidian Vault Spec

This spec defines how Obsidian notes must be structured to be indexed and retrieved
optimally by the House of Knowledge app.

---

## How the app processes notes

Every `.md` file is processed as follows:

1. **Frontmatter** is read for metadata — files without it are **skipped entirely**
2. The body is split into chunks on `##` and `###` headings — files with no such headings are **skipped entirely**
3. Each chunk is embedded and stored with its metadata, then retrieved by semantic similarity at query time
4. The retrieved chunks are passed to the LLM as context, along with the user's question

---

## Required frontmatter

Every note that should be searchable **must** have a frontmatter block. Files with no
`---` block are silently skipped — no error is shown in the app.

```yaml
---
session: 42
date: 2025-03-04
arc: Neverwinter
tags: [combat, npc, lord-neverember]
---
```

| Field     | Type            | Required | Notes                                                    |
|-----------|-----------------|----------|----------------------------------------------------------|
| `session` | integer         | Yes      | Used for filtering and citation display                  |
| `date`    | `YYYY-MM-DD`    | Yes      | Shown in source citations                                |
| `arc`     | string          | Yes      | Used for arc filter; must be consistent across sessions  |
| `tags`    | list of strings | Yes      | Used for tag filter; use lowercase, no spaces            |

For non-session pages (NPCs, locations, factions), use empty strings for `session` and
`date` — but the frontmatter block must still be present.

```yaml
---
session: ""
date: ""
arc: Neverwinter
tags: [npc, faction]
---
```

---

## Required body structure

The body must contain at least one `##` or `###` heading. Content before the first
heading is ignored by the indexer. Files with only a `#` (H1) heading, or no headings
at all, are skipped.

Each `##` or `###` section becomes one independently-searchable chunk. Chunks are the
unit of retrieval — keep them focused on a single topic.

### Example session note

```markdown
---
session: 42
date: 2025-03-04
arc: Neverwinter
tags: [combat, npc, lord-neverember]
---

## Summary

The party was summoned to [[Lord Neverember]]'s palace and given an ultimatum.
Combat broke out when [[Serath]] refused to kneel.

## Lord Neverember's Ultimatum

Neverember demanded the party retrieve the Seal of the Protector within ten days,
or face imprisonment. He was accompanied by two [[Mintarn Mercenary|Mintarn mercenaries]].

### The Terms

- Return the Seal by Session 43
- Do not enter the Blacklake District
- Swear a public oath of loyalty

## Combat in the Throne Room

[[Serath]] attacked first, targeting the left mercenary. The party fled via the
servant's passage after [[Zara]] cast Fog Cloud.
```

---

## Writing good chunks

The model retrieves chunks by semantic similarity to the user's question. Write each
section as if the reader has no other context — a chunk must make sense in isolation.

**Good — self-contained:**
```markdown
## Lord Neverember's Role

[[Lord Neverember]] is the Lord Protector of Neverwinter. He controls the city guard
and has been extorting local merchants. The party first encountered him in Session 38
during the dockside riot.
```

**Bad — relies on surrounding context:**
```markdown
## His Role

He controls everything mentioned above and met the party earlier.
```

### Chunk length guidelines

| Length       | Effect                                              |
|--------------|-----------------------------------------------------|
| < 50 words   | Too sparse — may not embed meaningfully             |
| 50–300 words | Ideal — focused and retrievable                     |
| 300–600 words| Acceptable for dense lore                           |
| 600+ words   | Dilutes retrieval — split into sub-sections         |

---

## Naming and linking

- **File names** become the URL path on the Quartz site. Name files exactly as you
  want them linked: `Lord Neverember.md` → `[[Lord Neverember]]` → clickable link
- Use `[[Wikilinks]]` freely in body text — the app converts them to links automatically
- Subfolder paths are preserved in URLs: `npcs/Lord Neverember.md` →
  `https://…/npcs/Lord Neverember`
- Aliased links are supported: `[[Lord Neverember|Neverember]]` renders as "Neverember"

---

## Recommended file types

| Note type      | One file per… | Suggested tags              |
|----------------|---------------|-----------------------------|
| Session recap  | Session       | `session-recap`, arc name   |
| NPC profile    | Character     | `npc`, faction name         |
| Location       | Place         | `location`, region          |
| Faction        | Faction       | `faction`                   |
| Item / lore    | Topic         | `item`, `lore`              |

---

## Common mistakes

| Problem                              | Result               | Fix                                          |
|--------------------------------------|----------------------|----------------------------------------------|
| No frontmatter block                 | File skipped         | Add `---` block with required fields         |
| Content only under `#` (H1)         | File skipped         | Use `##` or `###` for all sections           |
| Content before first `##`           | Content ignored      | Move it under a heading                      |
| Inconsistent `arc` spelling          | Filter splits data   | Pick one spelling and use it everywhere      |
| Sections over 600 words              | Retrieval diluted    | Break into multiple `##` subsections         |
| Vague section titles                 | Poor retrieval match | Use specific titles: "Neverember's Ultimatum" not "Events" |
| Plain names instead of `[[links]]`  | No clickable links   | Use `[[Name]]` for all characters, places, factions, items |
