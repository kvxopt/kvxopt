import concurrent.futures
import sys
import unittest

import kvxopt
from kvxopt import base, cholmod, klu, matrix, solvers, spmatrix, umfpack


class TestConcurrency(unittest.TestCase):
    def test_gil_state_freethreaded(self):
        """Under a free-threaded interpreter, the GIL must remain disabled after importing kvxopt."""
        if hasattr(sys, "_is_gil_enabled"):
            # If sys._is_gil_enabled exists and the build is free-threaded, verify it is False
            import sysconfig

            is_freethreaded_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
            if is_freethreaded_build:
                self.assertFalse(
                    sys._is_gil_enabled(),
                    "GIL was unexpectedly enabled! C extension modules must declare Py_MOD_GIL_NOT_USED.",
                )

    def test_options_isolation(self):
        """solvers.options must be context-isolated across concurrent threads."""
        results = {}

        def worker(thread_id: int):
            # Each thread sets its own options
            solvers.options["maxiters"] = 10 + thread_id
            solvers.options["show_progress"] = thread_id % 2 == 0
            results[thread_id] = (
                solvers.options["maxiters"],
                solvers.options["show_progress"],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(16)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        for thread_id, (maxiters, show_progress) in results.items():
            self.assertEqual(maxiters, 10 + thread_id)
            self.assertEqual(show_progress, thread_id % 2 == 0)

    def test_concurrent_lp(self):
        """Concurrent LP solving across threads must produce optimal results without race conditions."""
        solvers.options["show_progress"] = False

        def solve_lp(scale: float):
            # minimize c^T x subject to G x <= h
            # c = [1.0, 1.0] * scale
            # G = [-1, 0; 0, -1; 1, 1]
            # h = [0; 0; scale]
            c = matrix([1.0 * scale, 1.0 * scale])
            G = matrix([[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
            h = matrix([0.0, 0.0, scale])
            res = solvers.lp(c, G, h)
            return res["status"], list(res["x"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(solve_lp, float(i + 1)): i for i in range(20)}
            for future in concurrent.futures.as_completed(futures):
                status, x = future.result()
                self.assertEqual(status, "optimal")
                self.assertAlmostEqual(x[0], 0.0, places=5)
                self.assertAlmostEqual(x[1], 0.0, places=5)

    def test_concurrent_qp(self):
        """Concurrent QP solving across threads."""
        solvers.options["show_progress"] = False

        def solve_qp(offset: float):
            # minimize (1/2) x^T P x + q^T x subject to G x <= h
            # P = [2, 0; 0, 2]
            # q = [-offset, -offset]
            # x* = [offset/2, offset/2]
            P = 2.0 * matrix([[1.0, 0.0], [0.0, 1.0]])
            q = matrix([-offset, -offset])
            G = matrix([[-1.0, 0.0], [0.0, -1.0]])
            h = matrix([0.0, 0.0])
            res = solvers.qp(P, q, G, h)
            return res["status"], list(res["x"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(solve_qp, float(i + 1)): i for i in range(16)}
            for future in concurrent.futures.as_completed(futures):
                status, x = future.result()
                self.assertEqual(status, "optimal")
                i = futures[future]
                offset = float(i + 1)
                self.assertAlmostEqual(x[0], offset / 2.0, places=4)
                self.assertAlmostEqual(x[1], offset / 2.0, places=4)

    def test_concurrent_sparse_solvers(self):
        """Concurrent sparse factorizations and solves with CHOLMOD, KLU, and UMFPACK."""
        # Matrix:
        # [ 4  -1   0 ]
        # [-1   4  -1 ]
        # [ 0  -1   4 ]
        I = [0, 1, 0, 1, 2, 1, 2]
        J = [0, 0, 1, 1, 1, 2, 2]
        V = [4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0]
        A = spmatrix(V, I, J, (3, 3))

        def solve_cholmod(val: float):
            b = matrix([val, 2.0 * val, 3.0 * val])
            b_copy = matrix(b)
            cholmod.linsolve(A, b_copy)
            # Verify A * x == b
            res = matrix(0.0, (3, 1))
            base.gemv(A, b_copy, res)
            return [res[k] - b[k] for k in range(3)]

        def solve_klu(val: float):
            b = matrix([val, 2.0 * val, 3.0 * val])
            b_copy = matrix(b)
            klu.linsolve(A, b_copy)
            res = matrix(0.0, (3, 1))
            base.gemv(A, b_copy, res)
            return [res[k] - b[k] for k in range(3)]

        def solve_umfpack(val: float):
            b = matrix([val, 2.0 * val, 3.0 * val])
            b_copy = matrix(b)
            umfpack.linsolve(A, b_copy)
            res = matrix(0.0, (3, 1))
            base.gemv(A, b_copy, res)
            return [res[k] - b[k] for k in range(3)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            f_chol = [executor.submit(solve_cholmod, float(i + 1)) for i in range(10)]
            f_klu = [executor.submit(solve_klu, float(i + 1)) for i in range(10)]
            f_umf = [executor.submit(solve_umfpack, float(i + 1)) for i in range(10)]

            for f in concurrent.futures.as_completed(f_chol + f_klu + f_umf):
                residuals = f.result()
                for r in residuals:
                    self.assertAlmostEqual(r, 0.0, places=6)

    def test_buffer_exports_thread_safety(self):
        """Concurrent memoryview exports on a shared matrix must be thread-safe."""
        shared_mat = matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))

        def read_buffer():
            views = [memoryview(shared_mat) for _ in range(50)]
            total = sum(v[0, 0] for v in views)
            del views
            return total

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_buffer) for _ in range(16)]
            for f in concurrent.futures.as_completed(futures):
                self.assertAlmostEqual(f.result(), 50.0)

        # After all views are released, resizing should be allowed
        shared_mat.size = (3, 2)
        self.assertEqual(shared_mat.size, (3, 2))

    def test_buffer_export_resizing_blocked(self):
        """Resizing a matrix with active buffer exports must raise BufferError."""
        mat = matrix([1.0, 2.0, 3.0, 4.0], (2, 2))
        view = memoryview(mat)
        with self.assertRaises(BufferError):
            mat.size = (4, 1)
        del view
        # Once view is deleted, resizing works
        mat.size = (4, 1)
        self.assertEqual(mat.size, (4, 1))

    def test_blas_thread_controls(self):
        """Test set_blas_num_threads and blas_threads context manager."""
        kvxopt.set_blas_num_threads(2)
        with kvxopt.blas_threads(1):
            pass
        with self.assertRaises(ValueError):
            kvxopt.set_blas_num_threads(0)


if __name__ == "__main__":
    unittest.main()
