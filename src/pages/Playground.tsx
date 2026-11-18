// Free Python with NumPy and matplotlib, for trying things outside the exercises.
import { useState } from "react";
import Editor from "../components/Editor.tsx";
import { Output } from "../components/Output.tsx";
import { runCode, stop, type RunResult } from "../py/runner.ts";
import { saveCode, savedCode } from "../lib/progress.ts";

const START = `import numpy as np
import matplotlib.pyplot as plt

# Gradient descent on f(w) = (w - 3)^2, step by step
w, lr = 0.0, 0.2
path = []
for step in range(15):
    grad = 2 * (w - 3)
    w -= lr * grad
    path.append(w)
print(f"w after 15 steps: {w:.4f}")

plt.plot(path, marker="o")
plt.axhline(3, linestyle="--")
plt.title("w moves towards the minimum at 3")
plt.xlabel("step")
plt.show()
`;

export default function Playground() {
  const [code, setCode] = useState(() => savedCode("playground") ?? START);
  const [busy, setBusy] = useState(false);
  const [r, setR] = useState<RunResult | null>(null);
  const [err, setErr] = useState("");
  const run = async () => {
    if (busy) return;
    setBusy(true); setErr("");
    try { setR(await runCode(code)); } catch (e) { setErr((e as Error).message); setR(null); } finally { setBusy(false); }
  };
  return (
    <main id="main" tabIndex={-1} className="page">
      <header className="page-head">
        <h1>Playground</h1>
        <p className="lede">Python with NumPy and matplotlib, running in your browser. Call <code>plt.show()</code> to draw a figure.</p>
      </header>
      <Editor value={code} onChange={(v) => { setCode(v); saveCode("playground", v); }} onSubmit={run} label="Playground code" minLines={18} />
      <div className="ex-bar">
        <button className="btn primary" onClick={run} disabled={busy}>{busy ? "Running…" : "Run"}<kbd>Ctrl ↵</kbd></button>
        {busy && <button className="btn ghost" onClick={stop}>Stop</button>}
        <span className="grow" />
        <button className="btn ghost" onClick={() => { if (confirm("Replace your code with the example?")) { setCode(START); saveCode("playground", null); } }}>Reset</button>
      </div>
      {err && <div className="report bad" role="status"><pre>{err}</pre></div>}
      {r && <div className="report" role="status"><Output r={r} /></div>}
    </main>
  );
}
