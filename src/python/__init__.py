"""
Python package for convex optimization 

CVXOPT is a free software package for convex optimization based on the 
Python programming language. It can be used with the interactive Python 
interpreter, on the command line by executing Python scripts, or 
integrated in other software via Python extension modules. Its main 
purpose is to make the development of software for convex optimization 
applications straightforward by building on Python's extensive standard 
library and on the strengths of Python as a high-level programming 
language.
""" 

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

__copyright__ = """Copyright (c) 2012-2026 M. Andersen and L. Vandenberghe.
Copyright (c) 2010-2011 L. Vandenberghe.
Copyright (c) 2004-2009 J. Dahl and L. Vandenberghe."""

__license__ = """This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

CVXOPT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import sys
if sys.platform == 'win32':
    import os
    extra_dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.libs')
    if os.path.isdir(extra_dll_dir):
        os.add_dll_directory(extra_dll_dir)

import kvxopt.base

def copyright():
    print(__copyright__)

def license():
    print(__license__)

def normal(nrows, ncols=1, mean=0.0, std=1.0):
    '''
    Randomly generates a matrix with normally distributed entries.

    normal(nrows, ncols=1, mean=0, std=1)
  
    PURPOSE
    Returns a matrix with typecode 'd' and size nrows by ncols, with
    its entries randomly generated from a normal distribution with mean
    m and standard deviation std.

    ARGUMENTS
    nrows     number of rows

    ncols     number of columns

    mean      approximate mean of the distribution
    
    std       standard deviation of the distribution
    '''

    try:    
        from kvxopt import gsl
    except:
        from kvxopt.base import matrix
        from random import gauss
        return matrix([gauss(mean, std) for k in range(nrows*ncols)],
                      (nrows,ncols), 'd' )
        
    return gsl.normal(nrows, ncols, mean, std)

def uniform(nrows, ncols=1, a=0, b=1):
    '''
    Randomly generates a matrix with uniformly distributed entries.
    
    uniform(nrows, ncols=1, a=0, b=1)

    PURPOSE
    Returns a matrix with typecode 'd' and size nrows by ncols, with
    its entries randomly generated from a uniform distribution on the
    interval (a,b).

    ARGUMENTS
    nrows     number of rows

    ncols     number of columns

    a         lower bound

    b         upper bound
    '''

    try:    
        from kvxopt import gsl
    except:
        from kvxopt.base import matrix
        from random import uniform
        return matrix([uniform(a, b) for k in range(nrows*ncols)],
                      (nrows,ncols), 'd' )

    return gsl.uniform(nrows, ncols, a, b)


def weibull(nrows, ncols=1, a=1, b=1):
    '''
    Randomly generates a matrix with weibull distributed entries.
    
    weibull(nrows, ncols=1, a=1, b=1)

    PURPOSE
    Returns a matrix with typecode 'd' and size nrows by ncols, with
    its entries randomly generated from a weibull distribution.

    ARGUMENTS
    nrows     number of rows

    ncols     number of columns

    a         scale parameter

    b         shape parameter
    '''

    try:
        from kvxopt import gsl
    except:
        from kvxopt.base import matrix
        from random import weibullvariate
        return matrix([weibullvariate(a, b) for k in range(nrows*ncols)],
                      (nrows, ncols), 'd')

    return gsl.weibull(nrows, ncols, a, b)

def setseed(val = 0):
    ''' 
    Sets the seed value for the random number generator.

    setseed(val = 0)
    
    ARGUMENTS
    value     integer seed.  If the value is 0, the current system time  
              is used. 
    '''    

    try:    
        from kvxopt import gsl
        gsl.setseed(val)
    except:
        from random import seed
        if val == 0: val = None
        seed(val)
        
 
def getseed():
    '''
    Returns the seed value for the random number generator.
    
    getseed()
    '''

    try:    
        from kvxopt import gsl
        return gsl.getseed()
    except:
        raise NotImplementedError("getseed() not installed (requires GSL)")
    

import sys
if sys.version_info[0] < 3:
    import __builtin__
    omax = __builtin__.max
    omin = __builtin__.min
else:
    omax = max
    omin = min
    from functools import reduce


def max(*args):
    ''' 
    Elementwise max for matrices.

    PURPOSE
    max(a1, ..., an) computes the elementwise max for matrices.  The arguments
    must be matrices of compatible dimensions,  and scalars.  The elementwise
    max of a matrix and a scalar is a new matrix where each element is 
    defined as max of a matrix element and the scalar.

    max(iterable)  where the iterator generates matrices and scalars computes
    the elementwise max between the objects in the iterator,  using the
    same conventions as max(a1, ..., an).
    '''    
    
    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']: 
        return +reduce(kvxopt.base.emax, *args)
    elif len(args) == 1 and type(args[0]) is kvxopt.base.matrix:
        return omax(args[0])
    elif len(args) == 1 and type(args[0]) is kvxopt.base.spmatrix:
        if len(args[0]) == mul(args[0].size):
            return omax(args[0])
        else:
            return omax(omax(args[0]), 0.0)
    else:
        return +reduce(kvxopt.base.emax, args)


def min(*args):
    ''' 
    Elementwise min for matrices.

    PURPOSE
    min(a1, ..., an) computes the elementwise min for matrices.  The arguments
    must be matrices of compatible dimensions,  and scalars.  The elementwise
    min of a matrix and a scalar is a new matrix where each element is 
    defined as min of a matrix element and the scalar.

    min(iterable)  where the iterator generates matrices and scalars computes
    the elementwise min between the objects in the iterator,  using the
    same conventions as min(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']: 
        return +reduce(kvxopt.base.emin, *args)
    elif len(args) == 1 and type(args[0]) is kvxopt.base.matrix:
        return omin(args[0])
    elif len(args) == 1 and type(args[0]) is kvxopt.base.spmatrix:
        if len(args[0]) == mul(args[0].size):
            return omin(args[0])
        else:
            return omin(omin(args[0]), 0.0)
    else:
        return +reduce(kvxopt.base.emin, args)

def mul(*args):
    ''' 
    Elementwise multiplication for matrices.

    PURPOSE
    mul(a1, ..., an) computes the elementwise multiplication for matrices.
    The arguments must be matrices of compatible dimensions,  and scalars.  
    The elementwise multiplication of a matrix and a scalar is a new matrix 
    where each element is 
    defined as the multiplication of a matrix element and the scalar.

    mul(iterable)  where the iterator generates matrices and scalars computes
    the elementwise multiplication between the objects in the iterator,  
    using the same conventions as mul(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']: 
        return +reduce(kvxopt.base.emul, *args)
    else:
        return +reduce(kvxopt.base.emul, args)

def div(*args):
    ''' 
    Elementwise division for matrices.

    PURPOSE
    div(a1, ..., an) computes the elementwise division for matrices.
    The arguments must be matrices of compatible dimensions,  and scalars.  
    The elementwise division of a matrix and a scalar is a new matrix 
    where each element is defined as the division between a matrix element 
    and the scalar.  

    div(iterable)  where the iterator generates matrices and scalars computes
    the elementwise division between the objects in the iterator,  
    using the same conventions as div(a1, ..., an).
    '''

    if len(args) == 1 and type(args[0]).__name__ in \
            ['list', 'tuple', 'xrange', 'range', 'generator']: 
        return +reduce(kvxopt.base.ediv, *args)
    else:
        return +reduce(kvxopt.base.ediv, args)

kvxopt.base.normal, kvxopt.base.uniform = normal, uniform
kvxopt.base.setseed, kvxopt.base.getseed = setseed, getseed
kvxopt.base.mul, kvxopt.base.div = mul, div

from kvxopt import printing
matrix_str    = printing.matrix_str_default
matrix_repr   = printing.matrix_repr_default
spmatrix_str  = printing.spmatrix_str_default
spmatrix_repr = printing.spmatrix_repr_default

from kvxopt.base import matrix, spmatrix, sparse, spdiag, sqrt, sin, cos, \
    tan, asin, acos, atan, conj, exp, log, norm

from contextlib import contextmanager
import ctypes
import os

def _iter_blas_cdlls():
    """Iterate over candidate ctypes CDLL objects providing BLAS/LAPACK runtime hooks."""
    if sys.platform != "win32":
        try:
            yield ctypes.CDLL(None)
        except Exception:
            pass
    try:
        from kvxopt import blas as _blas_mod
        if hasattr(_blas_mod, "__file__") and _blas_mod.__file__:
            yield ctypes.CDLL(_blas_mod.__file__)
    except Exception:
        pass
    candidates = (
        "libopenblas.so", "libopenblas.dylib", "libopenblas.dll", "openblas.dll",
        "libmkl_rt.so", "libmkl_rt.dylib", "mkl_rt.dll",
        "libgomp.so.1", "libomp.so", "libomp.dylib"
    )
    for name in candidates:
        try:
            yield ctypes.CDLL(name)
        except Exception:
            pass

def _get_runtime_thread_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for cdll in _iter_blas_cdlls():
        if "openblas" not in counts and hasattr(cdll, "openblas_get_num_threads"):
            try:
                counts["openblas"] = int(cdll.openblas_get_num_threads())
            except Exception:
                pass
        if "mkl" not in counts:
            for getter in ("mkl_get_max_threads", "MKL_Get_Max_Threads"):
                if hasattr(cdll, getter):
                    try:
                        counts["mkl"] = int(getattr(cdll, getter)())
                        break
                    except Exception:
                        pass
        if "omp" not in counts and hasattr(cdll, "omp_get_max_threads"):
            try:
                counts["omp"] = int(cdll.omp_get_max_threads())
            except Exception:
                pass
    return counts

def set_blas_num_threads(n: int) -> None:
    """Set the number of threads for OpenBLAS / MKL / OMP.

    Parameters
    ----------
    n : int
        Number of threads (must be positive).

    Notes
    -----
    Sets environment variables OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, and
    OMP_NUM_THREADS. Also invokes direct in-process runtime symbols if
    OpenBLAS, MKL, or OpenMP is loaded.
    """
    if n < 1:
        raise ValueError("number of threads must be positive")

    for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[env] = str(n)

    c_n = ctypes.c_int(n)
    for cdll in _iter_blas_cdlls():
        if hasattr(cdll, "openblas_set_num_threads"):
            try:
                cdll.openblas_set_num_threads(c_n)
            except Exception:
                pass
        for setter in ("MKL_Set_Num_Threads", "mkl_set_num_threads"):
            if hasattr(cdll, setter):
                try:
                    getattr(cdll, setter)(c_n)
                except Exception:
                    pass
        if hasattr(cdll, "omp_set_num_threads"):
            try:
                cdll.omp_set_num_threads(c_n)
            except Exception:
                pass

def get_blas_num_threads() -> int | None:
    """Get the currently configured number of BLAS threads if detectable."""
    counts = _get_runtime_thread_counts()
    if counts:
        return next(iter(counts.values()))
    for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        val = os.environ.get(env)
        if val and val.isdigit():
            return int(val)
    return None

@contextmanager
def blas_threads(n: int):
    """Context manager for temporarily setting BLAS threads."""
    env_vars = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    old_env = {v: os.environ.get(v) for v in env_vars}
    prev_counts = _get_runtime_thread_counts()
    set_blas_num_threads(n)
    try:
        yield
    finally:
        for runtime, prev_val in prev_counts.items():
            c_val = ctypes.c_int(prev_val)
            for cdll in _iter_blas_cdlls():
                if runtime == "openblas" and hasattr(cdll, "openblas_set_num_threads"):
                    try:
                        cdll.openblas_set_num_threads(c_val)
                    except Exception:
                        pass
                elif runtime == "mkl":
                    for setter in ("MKL_Set_Num_Threads", "mkl_set_num_threads"):
                        if hasattr(cdll, setter):
                            try:
                                getattr(cdll, setter)(c_val)
                            except Exception:
                                pass
                elif runtime == "omp" and hasattr(cdll, "omp_set_num_threads"):
                    try:
                        cdll.omp_set_num_threads(c_val)
                    except Exception:
                        pass
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

from kvxopt import solvers, blas, lapack

__all__ = [ 'blas', 'lapack', 'amd', 'umfpack', 'klu', 'cholmod', 'solvers',
    'modeling', 'printing', 'info', 'matrix', 'spmatrix', 'sparse', 
    'spdiag', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'conj',
    'exp', 'log', 'min', 'max', 'mul', 
    'div', 'normal', 'uniform', 'setseed', 'getseed', 'norm',
    'set_blas_num_threads', 'get_blas_num_threads', 'blas_threads']

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

