import os, json, pathlib, subprocess, shlex, time
from typing import Dict, Any, List
from openai import OpenAI

# === CONFIG ===
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # pick a strong, cost-efficient coder
WORKSPACE = pathlib.Path(os.getenv("WORKSPACE", "./workspace")).resolve()
ALLOWED_CMDS = {
    "pytest": "pytest -q",
    "unit": "python -m pytest -q",
    "ruff": "ruff .",
    "mypy": "mypy .",
    "pip_freeze": "python -m pip freeze"
}
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))

client = OpenAI()

# --- Tool handlers (sandboxed to WORKSPACE) ---
def rp(p: str) -> pathlib.Path:
    pth = (WORKSPACE / p).resolve()
    if not str(pth).startswith(str(WORKSPACE)):
        raise ValueError("Path escape blocked")
    return pth

def read_file(path: str) -> Dict[str, Any]:
    p = rp(path)
    text = p.read_text(encoding="utf-8")
    return {"ok": True, "path": str(p), "text": text, "size": len(text)}

def write_file(path: str, content: str) -> Dict[str, Any]:
    p = rp(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(p), "size": len(content)}

def list_files(path: str = ".") -> Dict[str, Any]:
    p = rp(path)
    out = []
    for q in p.rglob("*"):
        if q.is_file() and (WORKSPACE in q.parents or q == WORKSPACE):
            rel = str(q.relative_to(WORKSPACE))
            if not rel.startswith(".git/"):
                out.append(rel)
    return {"ok": True, "files": sorted(out)[:1000]}

def run_cmd(alias: str) -> Dict[str, Any]:
    if alias not in ALLOWED_CMDS:
        return {"ok": False, "error": f"Command '{alias}' not allowed"}
    cmd = ALLOWED_CMDS[alias]
    try:
        proc = subprocess.run(shlex.split(cmd), cwd=str(WORKSPACE),
                              capture_output=True, text=True, timeout=300)
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Schema for function-calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file to the workspace, creating directories as needed.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path","content"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files (relative paths) under a directory in the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a whitelisted dev command in the workspace (pytest, ruff, mypy, pip_freeze).",
            "parameters": {"type": "object", "properties": {"alias": {"type": "string"}}, "required": ["alias"]}
        }
    }
]

SYSTEM = """You are a meticulous software agent working in a REPO at ./workspace.
Goal: complete the user's task with minimal changes and solid tests.
Rules:
- ALWAYS plan briefly before edits.
- Use list_files to discover structure; read_file before modifying.
- When writing files, produce complete final file content.
- Use run_cmd to run tests/linters; iterate until green.
- When finished, summarize what changed and how to run/tests.
"""

def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "read_file":  return read_file(**args)
        if name == "write_file": return write_file(**args)
        if name == "list_files": return list_files(**args)
        if name == "run_cmd":    return run_cmd(**args)
        return {"ok": False, "error": f"Unknown tool {name}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_agent(task: str):
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task}
    ]

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2
        )
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        # If the model issued tool calls, execute them and feed results back
        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                result = call_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result)
                })
            continue  # let model react to tool results

        # No tool calls -> likely done
        print(msg.content or "")
        break
    else:
        print("Stopped after MAX_STEPS without explicit completion.")

if __name__ == "__main__":
    import argparse, os, sys

    ap = argparse.ArgumentParser(description="Local coding agent (OpenAI tool-calling)")
    # Accept zero or more tokens and join them to a single task string
    ap.add_argument("task", nargs="*", help="e.g., Add RSI indicator and tests to src/alpha/signals.py")
    ap.add_argument("--workspace", default=str(WORKSPACE), help="Workspace directory")
    args = ap.parse_args()

    # Resolve workspace (no need for 'global' here)
    WORKSPACE = pathlib.Path(args.workspace).resolve()

    # Build task string (prompt if none was provided)
    task = " ".join(args.task).strip()
    if not task:
        try:
            task = input("Enter agent task: ").strip()
        except EOFError:
            pass
    if not task:
        print("No task provided. Example:\n  python agent.py \"Refactor final_production_bot.py to add idempotent client_order_id and tests\"")
        sys.exit(2)

    # Sanity check: API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your environment before running.")
        sys.exit(1)

    run_agent(task)