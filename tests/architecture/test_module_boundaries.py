"""
tests/architecture/test_module_boundaries.py

What does this cover?
- Every top-level package under agent/ is a RAG module with a declared list
  of modules it may depend on (MODULE_DEPENDENCIES).
- The declared dependency graph has no cycles.
- Code in one module imports another module only through its public API
  (`from agent.registry import RegistryService`), never through an internal
  path (`from agent.registry.store import ...`).
- Code imports only the modules it declares.
- backend/ sits outside agent/ and holds only backend logic: it reaches RAG
  modules only through their public APIs, never imports an ML, parsing or
  vector library itself, and nothing in agent/ imports it.

Why parse imports with ast instead of importing the code?
- Function-level imports (lazy loading in backend/dependencies.py, connectors)
  count too. Importing the code would miss them and load ML models.

How to add a module:
- Create agent/<name>/ with an __init__.py that exports its public API,
  then add <name> to MODULE_DEPENDENCIES below.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent"
BACKEND_DIR = ROOT / "backend"

# module → modules it may import.
MODULE_DEPENDENCIES: dict[str, set[str]] = {
    "shared":     set(),
    "ingestion":  set(),
    "registry":   set(),
    "connectors": set(),
    "processing": {"ingestion"},
    "chunking":   {"ingestion", "processing"},
    "embeddings": {"shared", "ingestion"},
    "retrieval":  {"shared", "embeddings"},
    "llm":        {"shared"},
    "qa":         {"shared", "retrieval", "llm"},
    "pipeline":   {"ingestion", "processing", "chunking", "embeddings", "registry"},
    "documents":  {"embeddings", "registry", "pipeline"},
    "sync":       {"connectors", "pipeline", "registry"},
}

# Third-party packages that do RAG work (models, parsing, chunking, vectors).
# Only RAG modules may import them; the backend calls those modules instead.
RAG_LIBRARIES = {
    "torch", "transformers", "sentence_transformers", "FlagEmbedding",
    "ollama", "langchain_ollama", "langchain_text_splitters", "chonkie",
    "fitz", "pymupdf", "pymupdf4llm", "docling", "unstructured", "pytesseract",
    "docx", "pptx", "openpyxl", "ftfy", "langdetect", "tiktoken", "datasketch",
    "numpy", "pgvector",
}


def _module_of(path: Path) -> str:
    return path.relative_to(AGENT_DIR).parts[0]


def _agent_files() -> list[Path]:
    return sorted(
        p for p in AGENT_DIR.rglob("*.py")
        if "__pycache__" not in p.parts and p.parent != AGENT_DIR
    )


def _backend_files() -> list[Path]:
    return sorted(p for p in BACKEND_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _absolute_name(path: Path, node: ast.ImportFrom) -> str:
    """Resolve `from .x import y` to `<package>...x`, e.g. `agent.retrieval.x`."""
    if node.level == 0:
        return node.module or ""
    package = list(path.relative_to(AGENT_DIR.parent).parent.parts)
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imports(path: Path) -> list[tuple[int, str]]:
    """(line, absolute dotted name) for every import in the file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = _absolute_name(path, node)
            if name == "agent":
                # `from agent import registry` imports the registry package root
                names = [f"agent.{alias.name}" for alias in node.names]
            else:
                names = [name]
        else:
            continue
        found.extend((node.lineno, name) for name in names)
    return found


def _agent_imports(path: Path) -> list[tuple[int, list[str]]]:
    """(line, dotted name split into parts) for every `agent.*` import in the file."""
    found = []
    for lineno, name in _imports(path):
        parts = name.split(".")
        if parts[0] == "agent" and len(parts) > 1:
            found.append((lineno, parts))
    return found


def _violations() -> list[str]:
    problems = []
    for path in _agent_files():
        source = _module_of(path)
        allowed = MODULE_DEPENDENCIES.get(source, set())
        rel = path.relative_to(ROOT)
        for lineno, parts in _agent_imports(path):
            target = parts[1]
            if target == source:
                continue
            name = ".".join(parts)
            if target not in allowed:
                problems.append(f"{rel}:{lineno} {source} → {target} is not declared ({name})")
            elif len(parts) > 2:
                problems.append(
                    f"{rel}:{lineno} imports internal path {name}; "
                    f"import from agent.{target} instead"
                )
    return problems


class TestModuleBoundaries:

    def test_every_package_is_a_declared_module(self):
        packages = {
            p.name for p in AGENT_DIR.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        }
        assert packages == set(MODULE_DEPENDENCIES)

    def test_declared_dependencies_are_known_modules(self):
        for module, deps in MODULE_DEPENDENCIES.items():
            assert deps <= set(MODULE_DEPENDENCIES), module
            assert module not in deps, module

    def test_dependency_graph_has_no_cycles(self):
        visiting, done = set(), set()

        def visit(module: str, trail: list[str]) -> None:
            if module in done:
                return
            if module in visiting:
                pytest.fail(f"cycle: {' → '.join(trail + [module])}")
            visiting.add(module)
            for dep in sorted(MODULE_DEPENDENCIES[module]):
                visit(dep, trail + [module])
            visiting.discard(module)
            done.add(module)

        for module in sorted(MODULE_DEPENDENCIES):
            visit(module, [])

    def test_imports_respect_module_boundaries(self):
        problems = _violations()
        assert not problems, "\n".join(problems)


class TestBackendBoundaries:

    def test_backend_package_exists(self):
        assert (BACKEND_DIR / "__init__.py").exists()

    def test_agent_never_imports_backend(self):
        problems = [
            f"{path.relative_to(ROOT)}:{lineno} imports {name}"
            for path in _agent_files()
            for lineno, name in _imports(path)
            if name.split(".")[0] == "backend"
        ]
        assert not problems, "\n".join(problems)

    def test_backend_imports_agent_public_apis_only(self):
        problems = [
            f"{path.relative_to(ROOT)}:{lineno} imports internal path {'.'.join(parts)}; "
            f"import from agent.{parts[1]} instead"
            for path in _backend_files()
            for lineno, parts in _agent_imports(path)
            if len(parts) > 2
        ]
        assert not problems, "\n".join(problems)

    def test_backend_imports_no_rag_libraries(self):
        problems = [
            f"{path.relative_to(ROOT)}:{lineno} imports {name}"
            for path in _backend_files()
            for lineno, name in _imports(path)
            if name.split(".")[0] in RAG_LIBRARIES
        ]
        assert not problems, "\n".join(problems)


class TestImportScanner:
    """The boundary check is only as good as its import parser."""

    def _write(self, tmp_path, monkeypatch, rel: str, source: str) -> Path:
        monkeypatch.setattr(f"{__name__}.AGENT_DIR", tmp_path / "agent")
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

    def _scan(self, tmp_path, monkeypatch, rel: str, source: str):
        path = self._write(tmp_path, monkeypatch, rel, source)
        return [".".join(parts) for _, parts in _agent_imports(path)]

    def test_finds_function_level_imports(self, tmp_path, monkeypatch):
        names = self._scan(
            tmp_path, monkeypatch, "backend/x.py",
            "def f():\n    from agent.registry.store import S\n",
        )
        assert names == ["agent.registry.store"]

    def test_resolves_relative_imports(self, tmp_path, monkeypatch):
        names = self._scan(
            tmp_path, monkeypatch, "agent/retrieval/sub/x.py",
            "from ..models import A\nfrom . import b\n",
        )
        assert names == ["agent.retrieval.models", "agent.retrieval.sub"]

    def test_resolves_relative_imports_outside_agent(self, tmp_path, monkeypatch):
        path = self._write(
            tmp_path, monkeypatch, "backend/routes/x.py", "from ..auth import f\n",
        )
        assert [name for _, name in _imports(path)] == ["backend.auth"]

    def test_from_agent_import_module(self, tmp_path, monkeypatch):
        names = self._scan(
            tmp_path, monkeypatch, "backend/x.py", "from agent import registry\n",
        )
        assert names == ["agent.registry"]

    def test_ignores_third_party_imports(self, tmp_path, monkeypatch):
        names = self._scan(
            tmp_path, monkeypatch, "backend/x.py", "import os\nfrom fastapi import Depends\n",
        )
        assert names == []
