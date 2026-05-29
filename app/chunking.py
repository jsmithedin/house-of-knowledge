import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    heading: str
    text: str


def slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


def chunk_markdown(content: str, source_path: str) -> list[Chunk]:
    """Split markdown body (after frontmatter) on ## and ### headings."""
    # Strip YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]

    lines = content.splitlines()
    chunks: list[Chunk] = []
    current_heading: str | None = None
    current_level: int = 0
    current_lines: list[str] = []

    heading_re = re.compile(r"^(#{2,3})\s+(.+)$")

    def flush():
        nonlocal current_heading, current_lines
        if current_heading and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                slug = slugify(current_heading)
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_path}#{slug}",
                        heading=current_heading,
                        text=f"## {current_heading}\n\n{body}",
                    )
                )
        current_lines = []

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            current_level = len(m.group(1))
            current_heading = m.group(2).strip()
        elif current_heading is not None:
            current_lines.append(line)

    flush()
    return chunks
