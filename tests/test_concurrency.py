import contextvars
import os
import sys
import sysconfig
import threading
import time
import unittest
from importlib import import_module

import kvxopt
from kvxopt import (
    base,
    blas,
    cholmod,
    klu,
    lapack,
    matrix,
    solvers,
    spmatrix,
    umfpack,
)

# Extensions that are always built. Optional wrappers are imported when present
# so a module that forgets Py_MOD_GIL_NOT_USED is still caught.
_CORE_EXTENSIONS = (
    "kvxopt.base",
    "kvxopt.blas",
    "kvxopt.lapack",
    "kvxopt.amd",
    "kvxopt.cholmod",
    "kvxopt.klu",
    "kvxopt.umfpack",
    "kvxopt.misc_solvers",
)
_OPTIONAL_EXTENSIONS = (
    "kvxopt.fftw",
    "kvxopt.gsl",
    "kvxopt.glpk",
    "kvxopt.osqp",
    "kvxopt.dsdp",
    "kvxopt.gurobi",
)
_BLAS_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def _is_free_threaded() -> bool:
    value = sysconfig.get_config_var("Py_GIL_DISABLED")
    if not value:
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return False


def _spd_tridiagonal(n: int, diag: float = 4.0, off: float = -1.0):
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(diag)
        if i:
            rows.append(i)
            cols.append(i - 1)
            vals.append(off)
            rows.append(i - 1)
            cols.append(i)
            vals.append(off)
    return spmatrix(vals, rows, cols, (n, n))


def _residual(A, x, b):
    res = matrix(0.0, (A.size[0], 1))
    base.gemv(A, x, res)
    return [res[k] - b[k] for k in range(A.size[0])]


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        self._saved_options = solvers.options.copy()
        solvers.options["show_progress"] = False
        self._saved_env = {key: os.environ.get(key) for key in _BLAS_ENV}
        self._saved_blas = kvxopt.get_blas_num_threads()
        self._gsl_seed = None
        try:
            from kvxopt import gsl
        except ImportError:
            gsl = None
        if gsl is not None:
            self._gsl_seed = gsl.getseed()
            self._gsl = gsl
        else:
            self._gsl = None

    def tearDown(self):
        solvers.options.clear()
        solvers.options.update(self._saved_options)
        if isinstance(self._saved_blas, int) and self._saved_blas >= 1:
            kvxopt.set_blas_num_threads(self._saved_blas)
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # setseed(0) means "use the clock", so a zero seed cannot be restored.
        if self._gsl is not None and self._gsl_seed:
            self._gsl.setseed(self._gsl_seed)

    def _assert_gil_matches_build(self):
        if not hasattr(sys, "_is_gil_enabled"):
            return
        if _is_free_threaded():
            self.assertFalse(
                sys._is_gil_enabled(),
                "GIL was enabled. Extension modules must declare Py_MOD_GIL_NOT_USED.",
            )
        else:
            self.assertTrue(sys._is_gil_enabled())

    def _run_threads(self, workers, timeout=60):
        errors = []
        threads = []

        def wrap(fn):
            def target():
                try:
                    fn()
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=target)
            thread.start()
            threads.append(thread)

        for worker in workers:
            wrap(worker)
        for thread in threads:
            thread.join(timeout)
            self.assertFalse(thread.is_alive(), "worker thread did not finish")
        if errors:
            raise errors[0]

    def test_gil_state_freethreaded(self):
        """Importing kvxopt must not re-enable the GIL on a free-threaded build."""
        imported = []
        for name in _CORE_EXTENSIONS:
            import_module(name)
            imported.append(name)
            self._assert_gil_matches_build()
        for name in _OPTIONAL_EXTENSIONS:
            try:
                import_module(name)
            except Exception:
                continue
            imported.append(name)
            self._assert_gil_matches_build()
        self.assertGreaterEqual(len(imported), len(_CORE_EXTENSIONS))
        if _is_free_threaded():
            self.assertTrue(hasattr(sys, "_is_gil_enabled"))
            self.assertFalse(sys._is_gil_enabled())

    def test_options_mapping_api(self):
        """solvers.options implements the mapping operations the solvers use."""
        self.assertIs(solvers.options, kvxopt.coneprog.options)
        self.assertIs(solvers.options, kvxopt.cvxprog.options)

        solvers.options.clear()
        self.assertEqual(len(solvers.options), 0)
        self.assertNotIn("maxiters", solvers.options)
        self.assertIsNone(solvers.options.get("maxiters"))

        solvers.options["maxiters"] = 12
        solvers.options.update(abstol=1e-8, reltol=1e-6)
        self.assertEqual(solvers.options["maxiters"], 12)
        self.assertEqual(solvers.options.setdefault("feastol", 1e-7), 1e-7)
        self.assertEqual(solvers.options.setdefault("feastol", 1.0), 1e-7)
        self.assertEqual(
            set(solvers.options),
            {"maxiters", "abstol", "reltol", "feastol"},
        )
        self.assertIn("maxiters", repr(solvers.options))

        copied = solvers.options.copy()
        copied["abstol"] = 1.0
        self.assertEqual(solvers.options["abstol"], 1e-8)
        self.assertEqual(solvers.options.pop("reltol"), 1e-6)
        del solvers.options["abstol"]
        with self.assertRaises(KeyError):
            solvers.options.pop("abstol")
        self.assertNotIn("abstol", solvers.options)
        solvers.options.clear()
        self.assertEqual(list(solvers.options), [])

    def test_options_context_isolation(self):
        """A copied context keeps its own options, including values the solver reads."""
        c = matrix([1.0, 1.0])
        G = matrix([[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
        h = matrix([0.0, 0.0, 1.0])

        solvers.options.clear()
        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 40
        solvers.options["abstol"] = 1e-8

        def reject():
            solvers.options["maxiters"] = 0
            self.assertEqual(solvers.options["abstol"], 1e-8)
            with self.assertRaises(ValueError):
                solvers.lp(c, G, h)
            self.assertEqual(solvers.options["maxiters"], 0)

        contextvars.copy_context().run(reject)
        self.assertEqual(solvers.options["maxiters"], 40)
        self.assertEqual(solvers.options["abstol"], 1e-8)
        self.assertEqual(solvers.lp(c, G, h)["status"], "optimal")

    def test_options_isolation(self):
        """Threads keep the options they set, and the parent context stays put."""
        c = matrix([1.0, 1.0])
        G = matrix([[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
        h = matrix([0.0, 0.0, 1.0])

        solvers.options.clear()
        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 25
        solvers.options["abstol"] = 2e-8

        ready_bad = threading.Event()
        ready_good = threading.Event()
        go = threading.Event()
        results = {}

        def bad():
            solvers.options["maxiters"] = 0
            solvers.options["show_progress"] = False
            ready_bad.set()
            self.assertTrue(go.wait(timeout=30))
            try:
                solvers.lp(c, G, h)
                results["bad"] = ("no-error", solvers.options["maxiters"])
            except ValueError:
                results["bad"] = ("value-error", solvers.options["maxiters"])

        def good():
            solvers.options["maxiters"] = 100
            solvers.options["show_progress"] = False
            ready_good.set()
            self.assertTrue(go.wait(timeout=30))
            status = solvers.lp(c, G, h)["status"]
            results["good"] = (status, solvers.options["maxiters"])

        bad_thread = threading.Thread(target=bad)
        bad_thread.start()
        self.assertTrue(ready_bad.wait(timeout=30))
        good_thread = threading.Thread(target=good)
        good_thread.start()
        self.assertTrue(ready_good.wait(timeout=30))
        # Both threads have published their options. A shared dict would now
        # hold whichever write landed last.
        go.set()
        bad_thread.join(timeout=30)
        good_thread.join(timeout=30)
        self.assertFalse(bad_thread.is_alive())
        self.assertFalse(good_thread.is_alive())

        self.assertEqual(results["bad"], ("value-error", 0))
        self.assertEqual(results["good"], ("optimal", 100))
        self.assertEqual(solvers.options["maxiters"], 25)
        self.assertEqual(solvers.options["abstol"], 2e-8)

    def test_concurrent_lp(self):
        """Concurrent LP solves return the optimum for each problem."""
        def solve_lp(scale: float):
            c = matrix([1.0 * scale, 1.0 * scale])
            G = matrix([[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
            h = matrix([0.0, 0.0, scale])
            res = solvers.lp(c, G, h)
            self.assertEqual(res["status"], "optimal")
            self.assertAlmostEqual(res["x"][0], 0.0, places=5)
            self.assertAlmostEqual(res["x"][1], 0.0, places=5)

        self._run_threads([lambda s=float(i + 1): solve_lp(s) for i in range(12)])
        self._assert_gil_matches_build()

    def test_concurrent_qp(self):
        """Concurrent QP solves return the optimum for each problem."""
        def solve_qp(offset: float):
            P = 2.0 * matrix([[1.0, 0.0], [0.0, 1.0]])
            q = matrix([-offset, -offset])
            G = matrix([[-1.0, 0.0], [0.0, -1.0]])
            h = matrix([0.0, 0.0])
            res = solvers.qp(P, q, G, h)
            self.assertEqual(res["status"], "optimal")
            self.assertAlmostEqual(res["x"][0], offset / 2.0, places=4)
            self.assertAlmostEqual(res["x"][1], offset / 2.0, places=4)

        self._run_threads([lambda o=float(i + 1): solve_qp(o) for i in range(8)])

    def test_concurrent_sparse_solvers(self):
        """Each thread factorizes its own matrix with CHOLMOD, KLU, and UMFPACK."""
        def solve(solver, scale: float):
            A = _spd_tridiagonal(24, diag=4.0 + scale)
            b = matrix([scale * (k + 1) for k in range(24)])
            x = matrix(b)
            solver(A, x)
            for residual in _residual(A, x, b):
                self.assertAlmostEqual(residual, 0.0, places=6)

        workers = []
        for solver in (cholmod.linsolve, klu.linsolve, umfpack.linsolve):
            for i in range(6):
                workers.append(lambda s=solver, scale=float(i + 1): solve(s, scale))
        self._run_threads(workers)
        self._assert_gil_matches_build()

    def test_concurrent_sparse_errors_stay_on_their_thread(self):
        """A singular factorization must not disturb a successful solve."""
        good = _spd_tridiagonal(12)
        singular = spmatrix([1.0, 0.0], [0, 1], [0, 1], (2, 2))
        expected_errors = {}
        for solver in (cholmod.linsolve, klu.linsolve, umfpack.linsolve):
            with self.assertRaises(ArithmeticError) as caught:
                solver(singular, matrix([1.0, 1.0]))
            expected_errors[solver] = str(caught.exception)

        def solve_good(solver):
            b = matrix([float(k + 1) for k in range(good.size[0])])
            x = matrix(b)
            solver(good, x)
            for residual in _residual(good, x, b):
                self.assertAlmostEqual(residual, 0.0, places=6)

        def solve_singular(solver):
            with self.assertRaises(ArithmeticError) as caught:
                solver(singular, matrix([1.0, 1.0]))
            self.assertEqual(str(caught.exception), expected_errors[solver])

        workers = []
        for solver in (cholmod.linsolve, klu.linsolve, umfpack.linsolve):
            for _ in range(4):
                workers.append(lambda s=solver: solve_good(s))
                workers.append(lambda s=solver: solve_singular(s))
        self._run_threads(workers)

    def test_concurrent_amd_order(self):
        """AMD ordering is safe to run from several threads at once."""
        from kvxopt import amd

        A = _spd_tridiagonal(30)
        expected = [int(v) for v in amd.order(A)]
        self.assertEqual(sorted(expected), list(range(30)))

        def check():
            order = [int(v) for v in amd.order(A)]
            self.assertEqual(order, expected)
            other = _spd_tridiagonal(18, diag=6.0)
            other_order = [int(v) for v in amd.order(other)]
            self.assertEqual(sorted(other_order), list(range(18)))

        self._run_threads([check for _ in range(8)])

    def test_concurrent_blas_and_lapack(self):
        """BLAS and LAPACK kernels return the serial result when run together."""
        A_gemm = matrix([[1.0, 3.0], [2.0, 4.0]])
        B_gemm = matrix([[1.0, 0.0], [0.0, 1.0]])
        C_expected = matrix(0.0, (2, 2))
        blas.gemm(A_gemm, B_gemm, C_expected)
        expected_gemm = list(C_expected)

        A_gesv = matrix([[2.0, 1.0], [1.0, 3.0]])
        b_gesv = matrix([5.0, 8.0])
        lapack.gesv(matrix(A_gesv), b_gesv)
        expected_gesv = list(b_gesv)

        def check_gemm(scale: float):
            C = matrix(0.0, (2, 2))
            blas.gemm(A_gemm, scale * B_gemm, C)
            for got, ref in zip(C, expected_gemm):
                self.assertAlmostEqual(got, scale * ref, places=8)

        def check_gesv(scale: float):
            b = scale * matrix([5.0, 8.0])
            lapack.gesv(matrix(A_gesv), b)
            for got, ref in zip(b, expected_gesv):
                self.assertAlmostEqual(got, scale * ref, places=8)

        workers = []
        for i in range(6):
            scale = float(i + 1)
            workers.append(lambda s=scale: check_gemm(s))
            workers.append(lambda s=scale: check_gesv(s))
        self._run_threads(workers)
        self._assert_gil_matches_build()

    def test_python_runs_during_blas_gemm(self):
        """Another Python thread must make progress during a BLAS GEMM.

        On GIL-enabled builds this requires Py_BEGIN_ALLOW_THREADS. On
        free-threaded builds the same overlap is required because the
        interpreter has no GIL.
        """
        kvxopt.set_blas_num_threads(1)
        n = 720
        A = matrix(1.0, (n, n))
        B = matrix(1.0, (n, n))
        C = matrix(0.0, (n, n))

        stop = threading.Event()
        ticks = []
        lock = threading.Lock()

        def ticker():
            while not stop.is_set():
                with lock:
                    ticks.append(time.perf_counter())
                time.sleep(0.0002)

        thread = threading.Thread(target=ticker)
        thread.start()
        deadline = time.perf_counter() + 8.0
        saw = False
        try:
            while time.perf_counter() < deadline and not saw:
                start = time.perf_counter()
                blas.gemm(A, B, C)
                end = time.perf_counter()
                with lock:
                    snapshot = list(ticks)
                saw = any(start < stamp < end for stamp in snapshot)
        finally:
            stop.set()
            thread.join(timeout=5)
        self.assertTrue(saw, "Python thread made no progress during GEMM")
        self.assertAlmostEqual(C[0, 0], float(n), places=6)
        self._assert_gil_matches_build()

    def test_gsl_rng_is_thread_local(self):
        """Each thread has its own GSL seed and stream."""
        if self._gsl is None:
            self.skipTest("GSL not available")
        gsl = self._gsl

        def draw(seed: int):
            gsl.setseed(seed)
            return [float(gsl.uniform(1)[0]) for _ in range(20)]

        reference_a = draw(12345)
        reference_b = draw(98765)
        self.assertNotEqual(reference_a, reference_b)
        gsl.setseed(12345)
        self.assertEqual(draw(12345), reference_a)

        barrier = threading.Barrier(8)
        results = {}

        def worker(thread_id: int):
            barrier.wait(timeout=30)
            seed = 12345 if thread_id % 2 == 0 else 98765
            results[thread_id] = (seed, draw(seed))

        self._run_threads([lambda i=i: worker(i) for i in range(8)])
        for thread_id, (seed, values) in results.items():
            expected = reference_a if seed == 12345 else reference_b
            self.assertEqual(values, expected)

    def test_concurrent_fftw(self):
        """Concurrent FFTW transforms round-trip and keep the planner locked."""
        try:
            from kvxopt import fftw
        except ImportError:
            self.skipTest("FFTW not available")

        def roundtrip(scale: float):
            original = matrix(
                [
                    complex(scale, 0.0),
                    complex(scale + 1.0, 1.0),
                    complex(0.0, scale),
                    complex(1.0, -scale),
                ]
            )
            expected = list(original)
            x = matrix(original)
            fftw.dft(x)
            fftw.idft(x)
            for got, ref in zip(x, expected):
                self.assertAlmostEqual(got.real, ref.real, places=8)
                self.assertAlmostEqual(got.imag, ref.imag, places=8)

        self._run_threads([lambda s=float(i + 1): roundtrip(s) for i in range(8)])
        self._assert_gil_matches_build()

    def test_buffer_exports_thread_safety(self):
        """Concurrent memoryview exports on a shared matrix must be thread-safe."""
        samples = (
            matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3)),
            matrix([1, 2, 3, 4, 5, 6], (2, 3), "i"),
            matrix([1 + 0j, 2, 3, 4, 5, 6], (2, 3)),
        )
        for shared in samples:
            def read_buffer(mat=shared):
                views = [memoryview(mat) for _ in range(40)]
                try:
                    total = sum(v[0, 0] for v in views)
                except NotImplementedError:
                    # Complex 'Zd' buffers cannot be indexed through memoryview.
                    # Creating and releasing the views still exercises ob_exports.
                    total = 40 * mat[0, 0]
                del views
                return total

            totals = [None] * 8

            def run(index, mat=shared):
                totals[index] = read_buffer(mat)

            self._run_threads([lambda i=i: run(i) for i in range(8)])
            for total in totals:
                self.assertIsNotNone(total)
                got = total.real if hasattr(total, "real") else total
                ref = shared[0, 0]
                ref = ref.real if hasattr(ref, "real") else ref
                self.assertAlmostEqual(float(got), float(40 * ref))
            shared.size = (3, 2)
            self.assertEqual(shared.size, (3, 2))

    def test_buffer_export_resizing_blocked(self):
        """Resizing a matrix with active buffer exports must raise BufferError."""
        for mat in (
            matrix([1.0, 2.0, 3.0, 4.0], (2, 2)),
            matrix([1, 2, 3, 4], (2, 2), "i"),
            matrix([1 + 1j, 2, 3, 4], (2, 2)),
        ):
            view = memoryview(mat)
            with self.assertRaises(BufferError):
                mat.size = (4, 1)
            self.assertEqual(mat.size, (2, 2))
            del view
            mat.size = (4, 1)
            self.assertEqual(mat.size, (4, 1))

    def test_resize_blocked_while_other_thread_holds_view(self):
        """A view held on another thread keeps resize from succeeding."""
        mat = matrix([1.0, 2.0, 3.0, 4.0], (2, 2))
        hold = threading.Event()
        release = threading.Event()
        errors = []

        def holder():
            view = memoryview(mat)
            try:
                self.assertEqual(view[0, 0], 1.0)
                hold.set()
                self.assertTrue(release.wait(timeout=30))
            except Exception as exc:
                errors.append(exc)
            finally:
                del view

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(hold.wait(timeout=30))
        with self.assertRaises(BufferError):
            mat.size = (4, 1)
        self.assertEqual(mat.size, (2, 2))
        release.set()
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        mat.size = (4, 1)
        self.assertEqual(mat.size, (4, 1))

    def _assert_blas_env(self, n: int) -> int | None:
        for key in _BLAS_ENV:
            self.assertEqual(os.environ[key], str(n))
        return kvxopt.get_blas_num_threads()

    def test_blas_thread_controls(self):
        """BLAS thread controls set, nest, and restore the runtime and environment.

        Environment variables always follow the request. A BLAS library may
        clamp the in-process count to the CPUs it was built for, so the
        runtime value is checked only when two requests produce different
        counts.
        """
        for key in _BLAS_ENV:
            os.environ.pop(key, None)

        with self.assertRaises(ValueError):
            kvxopt.set_blas_num_threads(0)
        with self.assertRaises(ValueError):
            kvxopt.set_blas_num_threads(-3)
        for key in _BLAS_ENV:
            self.assertNotIn(key, os.environ)

        with kvxopt.blas_threads(4):
            seen_four = self._assert_blas_env(4)
            with kvxopt.blas_threads(1):
                seen_one = self._assert_blas_env(1)
            seen_four_again = self._assert_blas_env(4)
        for key in _BLAS_ENV:
            self.assertNotIn(key, os.environ)

        if (
            None not in (seen_four, seen_one, seen_four_again)
            and seen_four != seen_one
        ):
            self.assertEqual(seen_four, seen_four_again)
            self.assertEqual(seen_one, 1)
            self.assertGreater(seen_four, seen_one)
            self.assertLessEqual(seen_four, 4)

        kvxopt.set_blas_num_threads(2)
        seen_two = self._assert_blas_env(2)
        if (
            seen_two is not None
            and seen_four is not None
            and seen_one is not None
            and seen_four != seen_one
        ):
            self.assertGreaterEqual(seen_two, 1)
            self.assertLessEqual(seen_two, 2)

        with self.assertRaises(ValueError):
            with kvxopt.blas_threads(0):
                pass
        # A rejected change leaves the previous thread count in place.
        for key in _BLAS_ENV:
            self.assertEqual(os.environ[key], "2")
        self.assertEqual(kvxopt.get_blas_num_threads(), seen_two)


if __name__ == "__main__":
    unittest.main()
