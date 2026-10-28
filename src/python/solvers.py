"""
Convex optimization solvers.

conelp:   solves linear cone programs.
coneqp:   solves quadratic cone programs.
cp:       solves nonlinear convex problem.
cpl:      solves nonlinear convex problems with linear objectives.
gp:       solves geometric programs.
lp:       solves linear programs.
qp:       solves quadratic programs.
sdp:      solves semidefinite programs.
socp:     solves second-order cone programs.
options:  dictionary with customizable algorithm parameters.
"""
from __future__ import annotations

# Copyright 2012-2026 M. Andersen and L. Vandenberghe.
# Copyright 2010-2011 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
# 
# This file is part of CVXOPT.
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import contextvars
from collections.abc import MutableMapping
from typing import Any, Iterator
import kvxopt
from kvxopt.cvxprog import cp, cpl, gp 
from kvxopt.coneprog import conelp, lp, sdp, socp, coneqp, qp

_options_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_kvxopt_solver_options", default=None
)

class _ThreadSafeOptions(MutableMapping[str, Any]):
    """Thread-safe context-local options dictionary for KVXOPT solvers."""

    def _data(self) -> dict[str, Any]:
        d = _options_var.get()
        if d is None:
            d = {}
            _options_var.set(d)
        return d

    def __getitem__(self, key: str) -> Any:
        return self._data()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        d = dict(self._data())
        d[key] = value
        _options_var.set(d)

    def __delitem__(self, key: str) -> None:
        d = dict(self._data())
        del d[key]
        _options_var.set(d)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def copy(self) -> dict[str, Any]:
        return self._data().copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data().get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        d = dict(self._data())
        val = d.setdefault(key, default)
        _options_var.set(d)
        return val

    def update(self, *args: Any, **kwargs: Any) -> None:
        d = dict(self._data())
        d.update(*args, **kwargs)
        _options_var.set(d)

    def clear(self) -> None:
        _options_var.set({})

    def __repr__(self) -> str:
        return repr(self._data())

    def __contains__(self, key: object) -> bool:
        return key in self._data()

options: MutableMapping[str, Any] = _ThreadSafeOptions()
kvxopt.cvxprog.options = options
kvxopt.coneprog.options = options
__all__ = ['conelp', 'coneqp', 'lp', 'socp', 'sdp', 'qp', 'cp', 'cpl', 'gp']
