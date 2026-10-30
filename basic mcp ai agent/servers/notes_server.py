"""Notes MCP server: a SQLite-backed notebook.

Demonstrates all three server primitives:
  * Tools     - add_note, search_notes, list_notes, delete_note
  * Resources - notes://stats (static) and notes://note/{note_id} (template)
  * Prompts   - summarize_notes

Run standalone (stdio transport):  python servers/notes_server.py
The database path comes from the NOTES_DB env var (default: workspace/notes.db).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError

DB_PATH = Path(os.environ.get("NOTES_DB", Path(__file__).resolve().parent.parent / "workspace" / "notes.db"))

mcp = MCPServer(
    "notes",
    instructions="A personal notebook. Use add_note to save information, search_notes to find it.",
)


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS notes (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               title TEXT NOT NULL,
               body TEXT NOT NULL,
               tags TEXT NOT NULL DEFAULT '',
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    return conn


def _fmt(row: sqlite3.Row) -> str:
    tags = f" [{row['tags']}]" if row["tags"] else ""
    return f"#{row['id']} {row['title']}{tags}: {row['body']}"


# ----------------------------------------------------------------- tools
@mcp.tool()
def add_note(title: str, body: str, tags: str = "") -> str:
    """Save a new note. `tags` is an optional comma-separated string."""
    if not title.strip():
        raise ToolError("title must not be empty")
    with closing(_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO notes (title, body, tags) VALUES (?, ?, ?)",
            (title.strip(), body.strip(), tags.strip()),
        )
        return f"Saved note #{cur.lastrowid}: {title.strip()}"


@mcp.tool()
def search_notes(query: str, limit: int = 10) -> str:
    """Search notes by keyword in title, body or tags (case-insensitive)."""
    like = f"%{query}%"
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR body LIKE ? OR tags LIKE ? "
            "ORDER BY id DESC LIMIT ?",
            (like, like, like, max(1, min(limit, 50))),
        ).fetchall()
    if not rows:
        return f"No notes match '{query}'."
    return "\n".join(_fmt(r) for r in rows)


@mcp.tool()
def list_notes(limit: int = 20) -> str:
    """List the most recent notes."""
    with closing(_conn()) as conn:
        rows = conn.execute("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
    return "\n".join(_fmt(r) for r in rows) if rows else "No notes yet."


@mcp.tool()
def delete_note(note_id: int) -> str:
    """DESTRUCTIVE: permanently delete a note by id. The agent asks for human approval first."""
    with closing(_conn()) as conn, conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return f"Deleted note #{note_id}." if cur.rowcount else f"No note with id {note_id}."


# ------------------------------------------------------------- resources
@mcp.resource("notes://stats")
def stats() -> str:
    """Summary statistics about the notebook."""
    with closing(_conn()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        latest = conn.execute("SELECT title FROM notes ORDER BY id DESC LIMIT 1").fetchone()
    return f"total_notes={total}; latest={latest['title'] if latest else 'n/a'}"


@mcp.resource("notes://note/{note_id}")
def note_by_id(note_id: str) -> str:
    """Read a single note by id."""
    with closing(_conn()) as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (int(note_id),)).fetchone()
    if row is None:
        raise ResourceNotFoundError(f"note {note_id} not found")
    return _fmt(row)


# --------------------------------------------------------------- prompts
@mcp.prompt()
def summarize_notes(topic: str = "everything") -> str:
    """A reusable prompt template that asks the model to summarise the notebook."""
    return (
        f"Use the notes tools to find notes about {topic}. "
        "Then write a short summary grouped by theme, citing note ids."
    )


if __name__ == "__main__":
    mcp.run("stdio")
