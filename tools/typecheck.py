import ast
import pathlib
import subprocess
import sys
import tempfile

import kvxopt


def iter_annotations(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                nodes.append(node.returns)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation is not None:
                    nodes.append(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation is not None:
                nodes.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                nodes.append(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign):
            nodes.append(node.annotation)
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            nodes.append(node.value)
    return nodes


def forbidden_annotation(node: ast.AST) -> str | None:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in {"Any", "object"}:
            return child.id
        if (
            isinstance(child, ast.Attribute)
            and child.attr == "Any"
            and isinstance(child.value, ast.Name)
            and child.value.id == "typing"
        ):
            return "typing.Any"
    return None


def symlink_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    try:
        dst.symlink_to(src)
    except OSError:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    package_dir = pathlib.Path(kvxopt.__file__).resolve().parent

    installed_stubs = sorted(package_dir.glob("*.pyi"))
    typed_fixtures = sorted((repo_root / "tests" / "types").glob("*.py"))

    if not installed_stubs:
        print(f"error: no installed stubs found under {package_dir}", file=sys.stderr)
        return 2
    if not typed_fixtures:
        print("error: no fixture files found under tests/types", file=sys.stderr)
        return 2

    violations: list[tuple[pathlib.Path, int, str]] = []
    for path in [*installed_stubs, *typed_fixtures]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for annotation in iter_annotations(tree):
            label = forbidden_annotation(annotation)
            if label:
                violations.append((path, getattr(annotation, "lineno", 1), label))

    if violations:
        print("error: forbidden annotations detected (Any/object are not allowed):", file=sys.stderr)
        for path, line, label in sorted(set(violations), key=lambda item: (str(item[0]), item[1], item[2])):
            print(f"  - {path}:{line}: {label}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="kvxopt-pyrefly-") as tmp:
        tmp_path = pathlib.Path(tmp)
        stub_pkg = tmp_path / "kvxopt"
        stub_pkg.mkdir(parents=True, exist_ok=True)

        for stub in installed_stubs:
            symlink_or_copy(stub, stub_pkg / stub.name)

        command = [
            "pyrefly",
            "check",
            "--config",
            "pyproject.toml",
            "--search-path",
            str(tmp_path),
            "--output-format",
            "full-text",
            *[str(path) for path in sorted(stub_pkg.glob("*.pyi"))],
            *[str(path) for path in typed_fixtures],
        ]
        print("Running:", " ".join(command))
        return subprocess.run(command, cwd=repo_root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
