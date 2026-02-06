from types import ModuleType

optional_modules = [
    "kvxopt.gsl",
    "kvxopt.fftw",
    "kvxopt.glpk",
    "kvxopt.osqp",
    "kvxopt.dsdp",
    "kvxopt.gurobi",
]

for module_name in optional_modules:
    try:
        module = __import__(module_name, fromlist=["*"])
    except Exception:
        module = None

    if module is not None:
        _: ModuleType = module
