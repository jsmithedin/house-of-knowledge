

import re


def resolve_wikilinks(text: str, wiki_base_url: str) -> str:
    """Convert [[Page]] and [[Page|Label]] wikilinks to markdown links."""
    base = wiki_base_url.rstrip("/")

    def replace(m: re.Match) -> str:
        inner = m.group(1)
        if "|" in inner:
            page, label = inner.split("|", 1)
        else:
            page = label = inner
        url = f"{base}/{page.strip()}"
        return f"[{label.strip()}]({url})"

    return re.sub(r"\[\[([^\]]+)\]\]", replace, text)


def build_wiki_url(wiki_base_url: str, source_path: str) -> str:
    path = source_path.removesuffix(".md")
    return f"{wiki_base_url.rstrip('/')}/{path}"


def format_sources_section(wiki_base_url: str, chunks: list[dict]) -> str:
    lines = ["---", "**Sources:**"]
    seen: set[str] = set()
    for chunk in chunks:
        key = chunk["source_path"]
        if key in seen:
            continue
        seen.add(key)
        session = chunk.get("session", "?")
        heading = chunk.get("heading", "Unknown")
        url = build_wiki_url(wiki_base_url, chunk["source_path"])
        lines.append(f"- [Session {session} — {heading}]({url})")
    return "\n".join(lines)
