# Module 1: Basic MCP Server — Implementation

A basic MCP server that helps developers create better pull requests by analyzing
code changes and suggesting appropriate templates. Implements the three course
tools for `pr-agent`, following the Unit 3 design philosophy:

> **Give the AI assistant raw data and let it decide — don't hardcode rigid rules.**

Host: **opencode** (via its built-in MCP client).
Environment: **conda env `aisystem`**.

## Implemented Tools

| Tool                     | Role                | What it returns                                                     |
| ------------------------ | ------------------- | ------------------------------------------------------------------- |
| `analyze_file_changes` | Data collection     | git diff + stats + changed files (truncated to the 25k token limit) |
| `get_pr_templates`     | Resource management | all PR templates from`templates/` with content                    |
| `suggest_template`     | Decision-making     | the best matching template + reasoning for the assistant to customize      |

## Setup

```bash
conda activate aisystem
cd projects/mcp-proj1
pip install "mcp[cli]" pytest pytest-asyncio
```

## Register with opencode

Add the server to opencode's MCP config (`opencode.json` in the project root):

```json
{
  "mcp": {
    "pr-agent": {
      "type": "local",
      "command": ["/path/to/conda/envs/aisystem/bin/python", "/absolute/path/to/mcp-proj1/server.py"],
      "environment": {}
    }
  }
}
```

If the `aisystem` env is already active in the terminal, `python server.py` also
works for reload/testing. Verify the server loads:

```bash
python -c "import server; print('pr-agent MCP server OK')"
```

Test:

```bash
pytest test_server.py -v
```

---

## `server.py` (full implementation)

Replace the contents of `server.py` with the code below.

```python
#!/usr/bin/env python3
"""
Module 1: Basic MCP Server
Analyzes git changes and suggests PR templates for opencode.
"""

import json
import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("pr-agent")

# PR template directory (shared across all modules)
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"

# MCP caps tool responses around 25,000 tokens — clip diffs well below that.
MAX_DIFF_LINES = 500


def _get_working_dir() -> str:
    """Return the assistant's working directory (from MCP roots) or the process cwd.

    Git commands must run where the assistant is working, not where the server was
    started. The server's cwd is a safe fallback when roots aren't available
    (e.g. in unit tests or the Inspector).
    """
    try:
        context = mcp.get_context()
        roots_result = context.session.list_roots() if hasattr(context, "session") else None
        roots = list(roots_result.roots) if roots_result and roots_result.roots else []
        if roots:
            # FileUrl objects expose the path via `.path`
            return getattr(roots[0].uri, "path", None) or os.getcwd()
    except Exception:
        pass
    return os.getcwd()


@mcp.tool()
async def analyze_file_changes(
    base_branch: str = "main",
    include_diff: bool = True,
    max_diff_lines: int = MAX_DIFF_LINES,
) -> str:
    """Analyze git changes between base_branch...HEAD and return raw data.

    Gives the assistant the raw diff, summary statistics, and the list of changed
    files so it can reason about the nature of the change.

    Args:
        base_branch: Base branch to compare against (default: main)
        include_diff: Whether to include the actual diff content (default: true)
        max_diff_lines: Maximum diff lines to include before truncating
    """
    try:
        working_dir = _get_working_dir()

        # Summary statistics
        stats = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            cwd=working_dir,
        )

        # Changed files with status (A/M/D)
        name_status = subprocess.run(
            ["git", "diff", "--name-status", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            cwd=working_dir,
        )

        # Full diff with smart truncation to respect the token limit
        diff_output = ""
        total_lines = 0
        if include_diff:
            diff = subprocess.run(
                ["git", "diff", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                cwd=working_dir,
            )
            diff_output = diff.stdout
            diff_lines = diff_output.split("\n")
            total_lines = len(diff_lines)
            if len(diff_lines) > max_diff_lines:
                diff_output = "\n".join(diff_lines[:max_diff_lines])
                diff_output += (
                    f"\n\n... Output truncated. "
                    f"Showing {max_diff_lines} of {len(diff_lines)} lines ..."
                )

        files_changed = [
            line for line in name_status.stdout.strip().split("\n") if line
        ]

        return json.dumps({
            "status": "success",
            "base_branch": base_branch,
            "stats": stats.stdout.strip(),
            "files_changed": files_changed,
            "total_diff_lines": total_lines,
            "truncated": total_lines > max_diff_lines if include_diff else False,
            "diff": diff_output if include_diff else "Set include_diff=true to see the diff",
        })
    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Run this tool inside a git repository"})


@mcp.tool()
async def get_pr_templates() -> str:
    """List available PR templates with their content.

    Returns every template in the shared `templates/` directory so the assistant can
    pick and customize the most appropriate one.
    """
    try:
        templates = []
        if TEMPLATES_DIR.exists():
            for path in sorted(TEMPLATES_DIR.glob("*.md")):
                content = path.read_text(encoding="utf-8")
                # Name = file stem (bug, docs, feature, ...), title = first heading
                first_heading = ""
                for line in content.splitlines():
                    if line.startswith("#"):
                        first_heading = line.lstrip("#").strip()
                        break
                templates.append({
                    "filename": path.name,
                    "name": path.stem,
                    "title": first_heading or path.stem,
                    "content": content,
                })
        return json.dumps({
            "status": "success",
            "count": len(templates),
            "templates": templates,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "hint": f"TEMPLATES_DIR = {TEMPLATES_DIR}"})


@mcp.tool()
async def suggest_template(changes_summary: str, change_type: str) -> str:
    """Suggest the most appropriate PR template for a change.

    Matches change_type to a template filename (bug, feature, docs, refactor,
    performance, security, test). The recommendation is a *suggestion*: the
    template content + changes_summary are returned so the assistant can verify the
    match and customize the template — the assistant stays in the loop.

    Args:
        changes_summary: Your analysis of what the changes do
        change_type: The type of change identified
            (bug, feature, docs, refactor, performance, security, test)
    """
    try:
        # Normalize aliases -> template filename
        aliases = {
            "fix": "bug", "bugfix": "bug", "bug": "bug",
            "feature": "feature", "feat": "feature",
            "docs": "docs", "doc": "docs", "documentation": "docs",
            "refactor": "refactor", "refactoring": "refactor",
            "performance": "performance", "perf": "performance",
            "security": "security", "sec": "security",
            "test": "test", "tests": "test", "testing": "test",
        }
        template_name = aliases.get(change_type.strip().lower(), change_type.strip().lower())

        templates_result = json.loads(await get_pr_templates())
        if templates_result.get("status") != "success":
            return json.dumps(templates_result)

        templates = templates_result["templates"]

        # Prefer exact template name; otherwise return all for the assistant to choose.
        recommended = next((t for t in templates if t["name"] == template_name), None)

        if recommended:
            return json.dumps({
                "status": "success",
                "recommended_template": recommended["filename"],
                "reason": f"change_type '{change_type}' maps directly to {recommended['filename']}",
                "template_content": recommended["content"],
                "changes_summary": changes_summary,
                "next_steps": "Customize the template content with the changes summary and draft the PR body.",
            })

        return json.dumps({
            "status": "success",
            "recommended_template": None,
            "reason": f"No template exactly matches change_type '{change_type}'. "
                      f"Choose the closest match below.",
            "available_templates": templates,
            "changes_summary": changes_summary,
            "next_steps": "Pick the closest template, load its content, and customize it.",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
```

---

## Design Notes

- **Raw data for the assistant.** `analyze_file_changes` returns the diff, `--stat`
  summary, and `--name-status` list; it does *not* classify the change. The assistant
  decides whether it's a bug fix, feature, refactor, etc.
- **Roots.** `_get_working_dir()` reads the assistant's working directory via
  `mcp.get_context()` → `session.list_roots()` and passes it as `cwd` to every
  git call. Falls back to the server's cwd so tests and the Inspector keep
  working.
- **Token limit.** Truncation (`max_diff_lines=500`, default) keeps responses
  well under the ~25k token response cap and reports `truncated` + total line
  count.
- **Always valid JSON**, including error paths — the assistant can parse any response.
- **Suggest, don't decide.** `suggest_template` returns the recommended
  template *and* its content plus the changes summary. The final call on
  template choice and wording stays with the assistant.

## Example workflow in opencode

```
User: "Analyze my changes and suggest a PR template."

Assistant:
1. Calls analyze_file_changes → raw diff + stats + changed files
2. Reasons about the kind of change (e.g. a bug in auth flow)
3. Calls get_pr_templates → sees available templates
4. Calls suggest_template("Fixed missing session validation in the login
   endpoint", "bug") → bug.md content
5. Drafts the PR using the bug.md template, customized to the change
```
