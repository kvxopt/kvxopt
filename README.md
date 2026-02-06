KVXOPT
======

This package is a fork from CVXOPT including more SuiteSparse functions and KLU
sparse matrix solver.


* [Website](https://sanurielf.github.io/kvxopt/)
* [Documentation](https://sanurielf.github.io/kvxopt/userguide/index.html)



Release info
------------

[![License](https://img.shields.io/badge/license-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html) 
[![GitHub release](https://img.shields.io/github/release/sanurielf/kvxopt.svg)](https://github.com/sanurielf/kvxopt/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/kvxopt.svg)](https://pypi.python.org/pypi/kvxopt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kvxopt/badges/version.svg)](https://anaconda.org/conda-forge/kvxopt)

Related projects
----------------

* [Original CVXOPT package](https://github.com/cvxopt/cvxopt)


Build status
------------


|             | [master](https://github.com/sanurielf/cvxopt/tree/master) | [dev](https://github.com/sanurielf/cvxopt/tree/dev) |
|-------------|--------|------------|
| Linux | [![Linux build](https://github.com/sanurielf/kvxopt/workflows/Linux%20build/badge.svg?branch=master)](https://github.com/sanurielf/kvxopt/actions)|  [![Linux build](https://github.com/sanurielf/kvxopt/workflows/Linux%20build/badge.svg?branch=dev)](https://github.com/sanurielf/kvxopt/actions)| 
| MacOs | [![macOS build](https://github.com/sanurielf/kvxopt/workflows/macOS%20build/badge.svg?branch=master)](https://github.com/sanurielf/kvxopt/actions)|  [![macOS build](https://github.com/sanurielf/kvxopt/workflows/macOS%20build/badge.svg?branch=dev)](https://github.com/sanurielf/kvxopt/actions)| 
| Windows (MSVC)| [![Windows build with MSVC](https://github.com/sanurielf/kvxopt/workflows/Windows%20build%20with%20MSVC/badge.svg?branch=master)](https://github.com/sanurielf/kvxopt/actions)|  [![Windows build with MSVC](https://github.com/sanurielf/kvxopt/workflows/Windows%20build%20with%20MSVC/badge.svg?branch=dev)](https://github.com/sanurielf/kvxopt/actions)| 
| Coveralls   |  [![Coverage Status](https://coveralls.io/repos/github/sanurielf/cvxopt/badge.svg?branch=master)](https://coveralls.io/github/sanurielf/cvxopt?branch=master)   | [![Coverage Status](https://coveralls.io/repos/github/sanurielf/cvxopt/badge.svg?branch=master)](https://coveralls.io/github/sanurielf/cvxopt?branch=dev)   |
| Readthedocs | [![RTFD Status](https://readthedocs.org/projects/cvxopt/badge/?version=latest)](http://cvxopt.readthedocs.io/en/latest/?badge=latest) | [![RTFD Status](https://readthedocs.org/projects/cvxopt/badge/?version=latest)](http://cvxopt.readthedocs.io/en/latest/?badge=latest) |

Type checking
-------------

`kvxopt` ships PEP 561 type information (`py.typed` + `.pyi` stubs) for the public
API, including C-extension modules.

The typing strategy is:
- stub-first for C-backed and highly dynamic modules;
- targeted inline annotations for small pure-Python modules;
- precise optional typing for extensions that may not be built (`gsl`, `fftw`,
  `glpk`, `osqp`, `dsdp`, `gurobi`).

Run static checks locally with:

```bash
pip install . pyrefly
python - <<'PY'
import ast
import pathlib
import subprocess
import sys
import tempfile

import kvxopt

pkg_dir = pathlib.Path(kvxopt.__file__).resolve().parent
stubs = sorted(pkg_dir.glob("*.pyi"))
fixtures = sorted(pathlib.Path("tests/types").glob("*.py"))

forbidden = {"Any", "object"}
violations = []
for path in [*stubs, *fixtures]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        for ann in (
            [node.annotation] if isinstance(node, ast.AnnAssign) else
            [node.returns, *(a.annotation for a in node.args.args + node.args.kwonlyargs if a.annotation)]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else
            []
        ):
            if ann is None:
                continue
            for child in ast.walk(ann):
                if isinstance(child, ast.Name) and child.id in forbidden:
                    violations.append((path, getattr(ann, "lineno", 1), child.id))
if violations:
    for path, line, label in sorted(set(violations), key=lambda item: (str(item[0]), item[1], item[2])):
        print(f"{path}:{line}: forbidden annotation {label}", file=sys.stderr)
    raise SystemExit(1)

with tempfile.TemporaryDirectory(prefix="kvxopt-pyrefly-") as tmp:
    root = pathlib.Path(tmp)
    stub_pkg = root / "kvxopt"
    stub_pkg.mkdir(parents=True, exist_ok=True)
    for stub in stubs:
        try:
            (stub_pkg / stub.name).symlink_to(stub)
        except OSError:
            (stub_pkg / stub.name).write_text(stub.read_text(encoding="utf-8"), encoding="utf-8")
    cmd = [
        "pyrefly", "check",
        "--config", "pyproject.toml",
        "--search-path", str(root),
        "--output-format", "full-text",
        *[str(path) for path in sorted(stub_pkg.glob("*.pyi"))],
        *[str(path) for path in fixtures],
    ]
    raise SystemExit(subprocess.run(cmd).returncode)
PY
```
