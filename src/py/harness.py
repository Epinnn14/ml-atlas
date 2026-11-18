# Runs learner code and exercise tests inside Pyodide. Shared by the browser worker and the
# Node content checker, so a solution that passes in CI passes in the browser too.
import base64
import io
import json
import sys
import time
import traceback

_figures = []


def _setup_matplotlib(code):
    """Make plt.show() hand PNGs back to the page instead of opening a window."""
    if "matplotlib" not in code:
        return
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    def show(*_a, **_k):
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            _figures.append(base64.b64encode(buf.getvalue()).decode())
        plt.close("all")

    plt.show = show


def _approx(a, b, tol=1e-6):
    """True when a and b match element-wise within tol (absolute or relative)."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol, rtol=tol))


def _short_trace(_kind):
    """Traceback limited to the learner's own lines."""
    etype, value, tb = sys.exc_info()
    frames = [f for f in traceback.extract_tb(tb) if f.filename in ("<your code>", "<tests>")]
    lines = [f'  line {f.lineno} in {f.name}: {(f.line or "").strip()}' for f in frames]
    head = f"{etype.__name__}: {value}"
    return head + ("\n" + "\n".join(lines) if lines else "")


def _exec_user(code, ns):
    _setup_matplotlib(code)
    exec(compile(code, "<your code>", "exec"), ns)
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None and plt.get_fignums():  # figures drawn without plt.show()
        plt.show()


def run_code(code):
    """Run code like a script: stdout, the error if any, and any figures shown."""
    _figures.clear()
    out = io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = out
    ns = {"__name__": "__main__"}
    error = None
    t0 = time.perf_counter()
    try:
        _exec_user(code, ns)
    except BaseException:
        error = _short_trace("run")
    finally:
        sys.stdout, sys.stderr = old
    return json.dumps({"stdout": out.getvalue()[-20000:], "error": error, "figures": list(_figures),
                       "ms": round((time.perf_counter() - t0) * 1000)})


def run_tests(code, tests):
    """Run the learner's code, then every test_* function from the tests, one result each."""
    _figures.clear()
    out = io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = out
    ns = {"__name__": "__main__"}
    results = []
    error = None
    try:
        _exec_user(code, ns)
    except BaseException:
        error = _short_trace("code")
    if error is None:
        tns = dict(ns)
        tns["approx"] = _approx
        before = set(tns)
        try:
            exec(compile(tests, "<tests>", "exec"), tns)
        except BaseException:
            error = "The tests could not start: " + _short_trace("tests")
        if error is None:
            names = [k for k in tns if k.startswith("test_") and k not in before and callable(tns[k])]
            for name in names:
                fn = tns[name]
                label = (fn.__doc__ or name[5:].replace("_", " ")).strip()
                try:
                    fn()
                    results.append({"name": label, "ok": True})
                except AssertionError as e:
                    results.append({"name": label, "ok": False, "message": str(e) or "the check did not hold"})
                except BaseException:
                    results.append({"name": label, "ok": False, "message": _short_trace("test")})
    sys.stdout, sys.stderr = old
    return json.dumps({"stdout": out.getvalue()[-20000:], "error": error, "results": results})
