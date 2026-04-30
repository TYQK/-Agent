#!/usr/bin/env python3
"""
Repo Review Agent
-----------------
A single-file, runnable code-review/refactor agent.

What it does:
1. Scans a repository for source files.
2. Runs rule-based review agents for Python/JS/TS code.
3. Optionally calls an OpenAI-compatible Chat Completions endpoint for deeper review.
4. Applies safe refactors when --fix is used.
5. Runs validation commands.
6. Writes a Markdown report, optional JSON report, and optional git diff patch.

No third-party Python dependencies are required.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclasses.dataclass
class Finding:
    severity: str              # critical | high | medium | low | info
    category: str              # security | maintainability | correctness | style | testing | duplication | llm
    file: str
    line: int
    message: str
    suggestion: str = ""
    source: str = "rule"       # rule | ast | duplicate | llm
    confidence: float = 0.85

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ValidationResult:
    command: str
    ok: bool
    exit_code: Optional[int]
    duration_seconds: float
    output_tail: str

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class AgentState:
    repo: Path
    files: List[Path] = dataclasses.field(default_factory=list)
    findings: List[Finding] = dataclasses.field(default_factory=list)
    changed_files: List[str] = dataclasses.field(default_factory=list)
    validations: List[ValidationResult] = dataclasses.field(default_factory=list)
    llm_tokens_estimated: int = 0
    started_at: str = dataclasses.field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


# -----------------------------
# Helpers
# -----------------------------

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".next", "coverage", ".idea", ".vscode",
}

CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".go", ".rs", ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".hpp",
}

TEXT_EXTS = CODE_EXTS | {".json", ".yaml", ".yml", ".toml", ".md", ".txt"}

SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|private[_-]?key)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
)

TODO_RE = re.compile(r"\b(TODO|FIXME|HACK)\b", re.IGNORECASE)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
    except Exception:
        return str(path).replace(os.sep, "/")


def read_text_safely(path: Path, max_bytes: int = 300_000) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def write_text_if_changed(path: Path, content: str) -> bool:
    old = read_text_safely(path, max_bytes=10_000_000)
    if old == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def compact_tail(text: str, limit: int = 3000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def approximate_tokens(text: str) -> int:
    # A rough planning estimate. English/code commonly averages around 3-5 chars/token.
    return max(1, len(text) // 4)


def is_git_repo(repo: Path) -> bool:
    return (repo / ".git").exists()


def run_subprocess(command: str, cwd: Path, timeout: int = 120) -> ValidationResult:
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = compact_tail(proc.stdout or "")
        return ValidationResult(
            command=command,
            ok=(proc.returncode == 0),
            exit_code=proc.returncode,
            duration_seconds=round(time.time() - start, 2),
            output_tail=output,
        )
    except subprocess.TimeoutExpired as exc:
        return ValidationResult(
            command=command,
            ok=False,
            exit_code=None,
            duration_seconds=round(time.time() - start, 2),
            output_tail=f"Timed out after {timeout}s. Partial output:\n{compact_tail(exc.stdout or '')}",
        )
    except Exception as exc:
        return ValidationResult(
            command=command,
            ok=False,
            exit_code=None,
            duration_seconds=round(time.time() - start, 2),
            output_tail=f"Failed to run command: {exc}",
        )


# -----------------------------
# Agent 1: scanner
# -----------------------------

class FileScannerAgent:
    def __init__(self, max_file_bytes: int = 300_000, max_files: int = 300):
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files

    def run(self, state: AgentState) -> AgentState:
        files: List[Path] = []
        for path in state.repo.rglob("*"):
            if len(files) >= self.max_files:
                break
            if path.is_dir():
                continue
            parts = set(path.relative_to(state.repo).parts)
            if parts & IGNORE_DIRS:
                continue
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except Exception:
                continue
            files.append(path)
        state.files = files
        return state


# -----------------------------
# Agent 2: deterministic rule reviewer
# -----------------------------

class RuleReviewerAgent:
    def run(self, state: AgentState) -> AgentState:
        for path in state.files:
            text = read_text_safely(path)
            if text is None:
                continue
            relative = rel(path, state.repo)
            self._line_rules(state, path, relative, text)
            if path.suffix == ".py":
                self._python_ast_rules(state, path, relative, text)
        self._duplicate_rules(state)
        return state

    def _add(self, state: AgentState, **kwargs: Any) -> None:
        state.findings.append(Finding(**kwargs))

    def _line_rules(self, state: AgentState, path: Path, relative: str, text: str) -> None:
        suffix = path.suffix.lower()
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            if SECRET_RE.search(line):
                self._add(
                    state,
                    severity="critical",
                    category="security",
                    file=relative,
                    line=idx,
                    message="Possible hardcoded credential or secret detected.",
                    suggestion="Move secrets to environment variables or a secret manager; rotate the exposed secret if real.",
                    source="rule",
                    confidence=0.95,
                )

            if len(line) > 140:
                self._add(
                    state,
                    severity="low",
                    category="style",
                    file=relative,
                    line=idx,
                    message=f"Line is very long ({len(line)} characters).",
                    suggestion="Split the expression, use named variables, or format the string across lines.",
                    source="rule",
                    confidence=0.80,
                )

            if TODO_RE.search(line):
                self._add(
                    state,
                    severity="info",
                    category="maintainability",
                    file=relative,
                    line=idx,
                    message="TODO/FIXME/HACK marker found.",
                    suggestion="Convert this into a tracked issue with owner, priority, and deadline.",
                    source="rule",
                    confidence=0.75,
                )

            if suffix == ".py":
                if re.match(r"^\s*except\s*:\s*(#.*)?$", line):
                    self._add(
                        state,
                        severity="medium",
                        category="correctness",
                        file=relative,
                        line=idx,
                        message="Bare except catches BaseException, including KeyboardInterrupt/SystemExit.",
                        suggestion="Use `except Exception:` or catch a specific exception type.",
                        source="rule",
                        confidence=0.95,
                    )
                if re.match(r"^\s*print\s*\(", line):
                    self._add(
                        state,
                        severity="low",
                        category="maintainability",
                        file=relative,
                        line=idx,
                        message="print() found in Python source.",
                        suggestion="Use the logging module for production code.",
                        source="rule",
                        confidence=0.70,
                    )
                if "eval(" in stripped or "exec(" in stripped:
                    self._add(
                        state,
                        severity="high",
                        category="security",
                        file=relative,
                        line=idx,
                        message="eval()/exec() can execute arbitrary code.",
                        suggestion="Replace with safe parsing, explicit dispatch tables, or ast.literal_eval where appropriate.",
                        source="rule",
                        confidence=0.90,
                    )
                if "shell=True" in line:
                    self._add(
                        state,
                        severity="high",
                        category="security",
                        file=relative,
                        line=idx,
                        message="subprocess call uses shell=True.",
                        suggestion="Pass arguments as a list and keep shell=False unless shell expansion is required and sanitized.",
                        source="rule",
                        confidence=0.88,
                    )

            if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                if re.search(r"\bconsole\.log\s*\(", line):
                    self._add(
                        state,
                        severity="low",
                        category="maintainability",
                        file=relative,
                        line=idx,
                        message="console.log() found.",
                        suggestion="Use a structured logger or remove debug logging before merging.",
                        source="rule",
                        confidence=0.75,
                    )
                if re.search(r"\bvar\s+", line):
                    self._add(
                        state,
                        severity="low",
                        category="style",
                        file=relative,
                        line=idx,
                        message="`var` declaration found.",
                        suggestion="Prefer `const` or `let` for block scoping.",
                        source="rule",
                        confidence=0.75,
                    )
                if re.search(r"(?<![=!])==(?!=)", line) or re.search(r"(?<![=!])!=(?!=)", line):
                    self._add(
                        state,
                        severity="medium",
                        category="correctness",
                        file=relative,
                        line=idx,
                        message="Loose equality found.",
                        suggestion="Use === or !== to avoid implicit coercion.",
                        source="rule",
                        confidence=0.80,
                    )
                if "eval(" in stripped:
                    self._add(
                        state,
                        severity="high",
                        category="security",
                        file=relative,
                        line=idx,
                        message="eval() can execute arbitrary code.",
                        suggestion="Use JSON.parse, explicit dispatch, or a safe interpreter for restricted expressions.",
                        source="rule",
                        confidence=0.90,
                    )

    def _python_ast_rules(self, state: AgentState, path: Path, relative: str, text: str) -> None:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            state.findings.append(Finding(
                severity="high",
                category="correctness",
                file=relative,
                line=exc.lineno or 1,
                message=f"Python syntax error: {exc.msg}",
                suggestion="Fix syntax before running automated refactors.",
                source="ast",
                confidence=0.99,
            ))
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                length = end - start + 1
                complexity = self._rough_complexity(node)
                if length > 80:
                    state.findings.append(Finding(
                        severity="medium",
                        category="maintainability",
                        file=relative,
                        line=start,
                        message=f"Function `{node.name}` is long ({length} lines).",
                        suggestion="Split it into smaller functions with single responsibilities and explicit tests.",
                        source="ast",
                        confidence=0.85,
                    ))
                if complexity > 14:
                    state.findings.append(Finding(
                        severity="medium",
                        category="maintainability",
                        file=relative,
                        line=start,
                        message=f"Function `{node.name}` has high rough complexity ({complexity}).",
                        suggestion="Reduce nested conditionals, extract branches, and add focused unit tests for edge cases.",
                        source="ast",
                        confidence=0.82,
                    ))

            if isinstance(node, ast.ExceptHandler):
                start = getattr(node, "lineno", 1)
                if node.type is not None:
                    name = self._exception_name(node.type)
                    if name in {"Exception", "BaseException"}:
                        state.findings.append(Finding(
                            severity="low" if name == "Exception" else "medium",
                            category="correctness",
                            file=relative,
                            line=start,
                            message=f"Broad exception handler catches `{name}`.",
                            suggestion="Catch the narrowest exception type you can handle correctly.",
                            source="ast",
                            confidence=0.80,
                        ))

    def _rough_complexity(self, node: ast.AST) -> int:
        score = 1
        decision_nodes = (
            ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.IfExp,
            ast.BoolOp, ast.Match, ast.ExceptHandler, ast.comprehension,
        )
        for child in ast.walk(node):
            if isinstance(child, decision_nodes):
                score += 1
        return score

    def _exception_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _duplicate_rules(self, state: AgentState) -> None:
        # Simple duplicate detector: normalized consecutive chunks of meaningful lines.
        # Capped to avoid report spam.
        chunk_index: Dict[str, Tuple[str, int]] = {}
        emitted = 0
        max_emitted = 25
        chunk_size = 8

        for path in state.files:
            if path.suffix.lower() not in CODE_EXTS:
                continue
            text = read_text_safely(path)
            if not text:
                continue
            relative = rel(path, state.repo)
            raw_lines = text.splitlines()
            normalized: List[Tuple[int, str]] = []
            for i, line in enumerate(raw_lines, start=1):
                s = line.strip()
                if not s or s.startswith(("#", "//", "*", "/*")):
                    continue
                s = re.sub(r"\s+", " ", s)
                normalized.append((i, s))

            for i in range(0, max(0, len(normalized) - chunk_size + 1)):
                chunk = "\n".join(x[1] for x in normalized[i:i + chunk_size])
                if len(chunk) < 80:
                    continue
                h = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                first_line = normalized[i][0]
                if h in chunk_index:
                    first_file, first_file_line = chunk_index[h]
                    if first_file != relative and emitted < max_emitted:
                        state.findings.append(Finding(
                            severity="medium",
                            category="duplication",
                            file=relative,
                            line=first_line,
                            message=f"Possible duplicate code block also seen in {first_file}:{first_file_line}.",
                            suggestion="Extract shared logic into a function/module or document why duplication is intentional.",
                            source="duplicate",
                            confidence=0.72,
                        ))
                        emitted += 1
                else:
                    chunk_index[h] = (relative, first_line)


# -----------------------------
# Agent 3: optional LLM reviewer
# -----------------------------

class LLMReviewerAgent:
    """Optional reviewer using an OpenAI-compatible chat completions endpoint.

    Environment variables:
      OPENAI_API_KEY or LLM_API_KEY
      LLM_BASE_URL, defaults to https://api.openai.com/v1/chat/completions
      LLM_MODEL, defaults to gpt-4o-mini

    This agent is optional. The script runs without any key in rule-only mode.
    """

    def __init__(self, model: str, base_url: str, api_key: Optional[str], max_files: int = 30):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_files = max_files

    def run(self, state: AgentState) -> AgentState:
        if not self.api_key:
            state.findings.append(Finding(
                severity="info",
                category="llm",
                file=".",
                line=1,
                message="LLM review skipped because OPENAI_API_KEY/LLM_API_KEY is not set.",
                suggestion="Set an API key or omit --llm to run rule-only mode.",
                source="llm",
                confidence=1.0,
            ))
            return state

        reviewed = 0
        candidate_files = [p for p in state.files if p.suffix.lower() in CODE_EXTS]
        # Review smaller files first to maximize coverage and control token cost.
        candidate_files.sort(key=lambda p: p.stat().st_size if p.exists() else 10**9)

        for path in candidate_files[: self.max_files]:
            text = read_text_safely(path, max_bytes=60_000)
            if not text:
                continue
            reviewed += 1
            relative = rel(path, state.repo)
            prompt = self._build_prompt(relative, text[:45_000])
            state.llm_tokens_estimated += approximate_tokens(prompt)
            try:
                items = self._call_llm_json(prompt)
            except Exception as exc:
                state.findings.append(Finding(
                    severity="info",
                    category="llm",
                    file=relative,
                    line=1,
                    message=f"LLM review failed for this file: {exc}",
                    suggestion="Check API key, model name, base URL, or run without --llm.",
                    source="llm",
                    confidence=1.0,
                ))
                continue
            for item in items[:8]:
                try:
                    severity = str(item.get("severity", "medium")).lower()
                    if severity not in SEVERITY_ORDER:
                        severity = "medium"
                    line = int(item.get("line", 1) or 1)
                    msg = str(item.get("message", "LLM finding")).strip()[:500]
                    suggestion = str(item.get("suggestion", "")).strip()[:500]
                    category = str(item.get("category", "llm")).strip().lower()[:40] or "llm"
                    state.findings.append(Finding(
                        severity=severity,
                        category=category,
                        file=relative,
                        line=max(1, line),
                        message=msg,
                        suggestion=suggestion,
                        source="llm",
                        confidence=0.70,
                    ))
                except Exception:
                    continue

        if reviewed == 0:
            state.findings.append(Finding(
                severity="info",
                category="llm",
                file=".",
                line=1,
                message="No eligible files were small enough for LLM review.",
                suggestion="Increase size limits in the code if you need deeper review.",
                source="llm",
                confidence=1.0,
            ))
        return state

    def _build_prompt(self, filename: str, code: str) -> str:
        return textwrap.dedent(f"""
        You are a senior software engineer doing a concise code review.
        Review this file and return ONLY valid JSON, with this schema:
        [
          {{
            "severity": "critical|high|medium|low|info",
            "category": "security|correctness|maintainability|testing|performance|style",
            "line": 123,
            "message": "specific issue",
            "suggestion": "specific fix"
          }}
        ]

        Rules:
        - Focus on real issues, not taste.
        - Prefer findings that could break production, cause security risks, or make maintenance hard.
        - Max 8 findings.
        - If no meaningful issues exist, return [].

        FILE: {filename}
        CODE:
        ```
        {code}
        ```
        """).strip()

    def _call_llm_json(self, prompt: str) -> List[Dict[str, Any]]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON. Do not use markdown."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {compact_tail(err, 800)}") from exc

        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
        content = content.strip()
        # Robustly extract JSON array even if the model accidentally adds text.
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        arr = json.loads(content[start:end + 1])
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
        return []


# -----------------------------
# Agent 4: safe patcher
# -----------------------------

class PatchAgent:
    """Applies conservative, low-risk transformations only."""

    def run(self, state: AgentState) -> AgentState:
        changed: List[str] = []
        for path in state.files:
            if path.suffix.lower() not in TEXT_EXTS:
                continue
            text = read_text_safely(path, max_bytes=1_000_000)
            if text is None:
                continue
            new_text = self._safe_format(text)
            if path.suffix == ".py":
                new_text = self._fix_python_bare_except(new_text)
            if new_text != text:
                if write_text_if_changed(path, new_text):
                    changed.append(rel(path, state.repo))
        state.changed_files = changed
        return state

    def _safe_format(self, text: str) -> str:
        lines = text.splitlines()
        # strip trailing whitespace, preserve content
        text = "\n".join(line.rstrip() for line in lines)
        # final newline is friendly to POSIX tools and git
        return text + "\n"

    def _fix_python_bare_except(self, text: str) -> str:
        # Preserves indentation and trailing comments.
        return re.sub(r"^(\s*)except\s*:(\s*(#.*)?)$", r"\1except Exception:\2", text, flags=re.MULTILINE)


# -----------------------------
# Agent 5: validation
# -----------------------------

class ValidationAgent:
    def __init__(self, commands: Sequence[str], include_auto: bool, timeout: int):
        self.commands = list(commands)
        self.include_auto = include_auto
        self.timeout = timeout

    def run(self, state: AgentState) -> AgentState:
        commands: List[str] = []
        if self.include_auto:
            commands.extend(self._auto_commands(state.repo, state.files))
        commands.extend(self.commands)

        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd and cmd not in seen:
                unique_commands.append(cmd)
                seen.add(cmd)

        for cmd in unique_commands:
            state.validations.append(run_subprocess(cmd, cwd=state.repo, timeout=self.timeout))
        return state

    def _auto_commands(self, repo: Path, files: Sequence[Path]) -> List[str]:
        commands: List[str] = []
        py_files = [rel(p, repo) for p in files if p.suffix == ".py"]
        if py_files:
            # Shell-quote file names so paths with spaces are safe.
            batches = [py_files[i:i + 80] for i in range(0, len(py_files), 80)]
            for batch in batches[:3]:
                commands.append(f"{shlex.quote(sys.executable)} -m py_compile " + " ".join(shlex.quote(x) for x in batch))
        if (repo / "pytest.ini").exists() or (repo / "pyproject.toml").exists() or (repo / "tests").exists():
            commands.append(f"{shlex.quote(sys.executable)} -m pytest -q")
        if (repo / "package.json").exists():
            commands.append("npm test --if-present")
        return commands


# -----------------------------
# Agent 6: report
# -----------------------------

class ReportAgent:
    def __init__(self, out_path: Path, json_out_path: Optional[Path] = None):
        self.out_path = out_path
        self.json_out_path = json_out_path

    def run(self, state: AgentState) -> AgentState:
        state.findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.file, f.line))
        report = self._markdown(state)
        self.out_path.write_text(report, encoding="utf-8")
        if self.json_out_path:
            payload = {
                "repo": str(state.repo),
                "started_at": state.started_at,
                "files_scanned": len(state.files),
                "changed_files": state.changed_files,
                "llm_tokens_estimated": state.llm_tokens_estimated,
                "findings": [f.as_dict() for f in state.findings],
                "validations": [v.as_dict() for v in state.validations],
            }
            self.json_out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def _markdown(self, state: AgentState) -> str:
        counts = {k: 0 for k in SEVERITY_ORDER}
        for f in state.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        lines: List[str] = []
        lines.append("# Repo Review Agent Report")
        lines.append("")
        lines.append(f"- Repository: `{state.repo}`")
        lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
        lines.append(f"- Files scanned: **{len(state.files)}**")
        lines.append(f"- Findings: **{len(state.findings)}**")
        lines.append(f"- Estimated LLM prompt tokens: **{state.llm_tokens_estimated}**")
        lines.append("")
        lines.append("## Summary by Severity")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|---|---:|")
        for sev in ["critical", "high", "medium", "low", "info"]:
            lines.append(f"| {sev} | {counts.get(sev, 0)} |")
        lines.append("")

        if state.changed_files:
            lines.append("## Safe Fixes Applied")
            lines.append("")
            for f in state.changed_files:
                lines.append(f"- `{f}`")
            lines.append("")

        lines.append("## Findings")
        lines.append("")
        if not state.findings:
            lines.append("No findings. Nice work.")
            lines.append("")
        else:
            for i, f in enumerate(state.findings, start=1):
                lines.append(f"### {i}. `{f.severity.upper()}` {f.category} — `{f.file}:{f.line}`")
                lines.append("")
                lines.append(f"**Message:** {f.message}")
                if f.suggestion:
                    lines.append("")
                    lines.append(f"**Suggestion:** {f.suggestion}")
                lines.append("")
                lines.append(f"Source: `{f.source}`, confidence: `{f.confidence:.2f}`")
                lines.append("")

        lines.append("## Validation")
        lines.append("")
        if not state.validations:
            lines.append("No validation commands were run. Use `--validate` or `--cmd \"your command\"`.")
            lines.append("")
        else:
            for v in state.validations:
                status = "PASS" if v.ok else "FAIL"
                lines.append(f"### `{status}` `{v.command}`")
                lines.append("")
                lines.append(f"- Exit code: `{v.exit_code}`")
                lines.append(f"- Duration: `{v.duration_seconds}s`")
                if v.output_tail:
                    lines.append("")
                    lines.append("```text")
                    lines.append(v.output_tail)
                    lines.append("```")
                lines.append("")

        lines.append("## Recommended Next Steps")
        lines.append("")
        lines.append("1. Fix critical/high findings first, especially secrets, unsafe eval/exec, shell=True, and syntax errors.")
        lines.append("2. For medium maintainability findings, split large functions and add tests around changed behavior.")
        lines.append("3. Re-run this agent with `--fix --validate` before opening a PR.")
        lines.append("4. In CI, run it in report-only mode first; after trust is built, allow safe formatting fixes.")
        lines.append("")
        return "\n".join(lines)


# -----------------------------
# Git patch export
# -----------------------------

def export_git_patch(repo: Path, patch_out: Path) -> Optional[str]:
    if not is_git_repo(repo):
        return "Not a git repository; skipped patch export."
    result = subprocess.run(
        "git diff -- .",
        cwd=str(repo),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    patch_out.write_text(result.stdout or "", encoding="utf-8")
    return None


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local code review/refactor agent over a repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default=".", help="Repository path to scan")
    parser.add_argument("--out", default="agent_review_report.md", help="Markdown report output path")
    parser.add_argument("--json-out", default="", help="Optional JSON report output path")
    parser.add_argument("--patch-out", default="", help="Optional git diff patch output path, useful with --fix")
    parser.add_argument("--max-files", type=int, default=300, help="Max files to scan")
    parser.add_argument("--fix", action="store_true", help="Apply safe fixes: trailing whitespace, final newline, Python bare except")
    parser.add_argument("--validate", action="store_true", help="Run auto-detected validation commands")
    parser.add_argument("--cmd", action="append", default=[], help="Extra validation command; can be repeated")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per validation command in seconds")
    parser.add_argument("--llm", action="store_true", help="Enable optional LLM review")
    parser.add_argument("--llm-max-files", type=int, default=30, help="Max files for LLM review")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="LLM model name")
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        help="OpenAI-compatible chat completions endpoint",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists() or not repo.is_dir():
        print(f"ERROR: repo path does not exist or is not a directory: {repo}", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser().resolve()
    json_out_path = Path(args.json_out).expanduser().resolve() if args.json_out else None
    patch_out_path = Path(args.patch_out).expanduser().resolve() if args.patch_out else None

    state = AgentState(repo=repo)

    print("[1/6] Scanning files...")
    FileScannerAgent(max_files=args.max_files).run(state)
    print(f"      scanned candidates: {len(state.files)}")

    print("[2/6] Running rule reviewer...")
    RuleReviewerAgent().run(state)
    print(f"      findings so far: {len(state.findings)}")

    if args.llm:
        print("[3/6] Running optional LLM reviewer...")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        LLMReviewerAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            max_files=args.llm_max_files,
        ).run(state)
        print(f"      findings after LLM: {len(state.findings)}")
    else:
        print("[3/6] LLM reviewer skipped. Use --llm to enable it.")

    if args.fix:
        print("[4/6] Applying safe fixes...")
        PatchAgent().run(state)
        print(f"      changed files: {len(state.changed_files)}")
    else:
        print("[4/6] Safe fixes skipped. Use --fix to apply them.")

    print("[5/6] Running validation...")
    ValidationAgent(commands=args.cmd, include_auto=args.validate, timeout=args.timeout).run(state)
    print(f"      validation commands: {len(state.validations)}")

    print("[6/6] Writing report...")
    ReportAgent(out_path=out_path, json_out_path=json_out_path).run(state)
    print(f"      markdown report: {out_path}")
    if json_out_path:
        print(f"      json report: {json_out_path}")

    if patch_out_path:
        msg = export_git_patch(repo, patch_out_path)
        if msg:
            print(f"      patch export: {msg}")
        else:
            print(f"      patch file: {patch_out_path}")

    high_risk = [f for f in state.findings if f.severity in {"critical", "high"}]
    failed_validations = [v for v in state.validations if not v.ok]
    print("")
    print("Done.")
    print(f"Findings: {len(state.findings)} total, {len(high_risk)} critical/high.")
    print(f"Validation: {len(state.validations) - len(failed_validations)}/{len(state.validations)} passed.")

    # CI-friendly exit code: fail only on critical findings or validation failure.
    if any(f.severity == "critical" for f in state.findings) or failed_validations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
