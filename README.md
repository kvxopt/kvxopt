KVXOPT
======

This package is a fork from CVXOPT including more SuiteSparse functions and KLU
sparse matrix solver.


* [Website](https://kvxopt.github.io/kvxopt/)
* [Documentation](https://kvxopt.github.io/kvxopt/userguide/index.html)



Release info
------------

[![License](https://img.shields.io/badge/license-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html) 
[![GitHub release](https://img.shields.io/github/release/kvxopt/kvxopt.svg)](https://github.com/kvxopt/kvxopt/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/kvxopt.svg)](https://pypi.python.org/pypi/kvxopt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/kvxopt/badges/version.svg)](https://anaconda.org/conda-forge/kvxopt)

Related projects
----------------

* [Original CVXOPT package](https://github.com/cvxopt/cvxopt)


Build status
------------


|             | [master](https://github.com/kvxopt/kvxopt/tree/master) | [dev](https://github.com/kvxopt/kvxopt/tree/dev) |
|-------------|--------|------------|
| Linux | [![Linux build](https://github.com/kvxopt/kvxopt/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/kvxopt/kvxopt/actions/workflows/build.yml) | [![Linux build](https://github.com/kvxopt/kvxopt/actions/workflows/build.yml/badge.svg?branch=dev)](https://github.com/kvxopt/kvxopt/actions/workflows/build.yml) |
| Wheels (Linux) | [![Build wheels (Linux)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_linux.yml/badge.svg?branch=master)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_linux.yml) | [![Build wheels (Linux)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_linux.yml/badge.svg?branch=dev)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_linux.yml) |
| Wheels (macOS) | [![Build wheels (macOS)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_macos.yml/badge.svg?branch=master)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_macos.yml) | [![Build wheels (macOS)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_macos.yml/badge.svg?branch=dev)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_macos.yml) |
| Wheels (Windows) | [![Build wheels (Windows)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_windows.yml/badge.svg?branch=master)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_windows.yml) | [![Build wheels (Windows)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_windows.yml/badge.svg?branch=dev)](https://github.com/kvxopt/kvxopt/actions/workflows/build_wheels_windows.yml) |
| Codecov | [![codecov](https://codecov.io/gh/kvxopt/kvxopt/branch/master/graph/badge.svg)](https://codecov.io/gh/kvxopt/kvxopt) | [![codecov](https://codecov.io/gh/kvxopt/kvxopt/branch/dev/graph/badge.svg)](https://codecov.io/gh/kvxopt/kvxopt) |

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
python tools/typecheck.py
```
