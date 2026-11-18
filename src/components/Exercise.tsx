// One coding exercise: prompt, editor, Run / Check, hints, the solution, and the test report.
import { useState } from "react";
import type { Exercise as Ex } from "../lib/lesson.ts";
import { runCode, runTests, stop, usePyStatus, type RunResult, type TestResult } from "../py/runner.ts";
import { markPassed, markPeeked, saveCode, savedCode, useProgress } from "../lib/progress.ts";
import Editor from "./Editor.tsx";
import Markdown from "./Markdown.tsx";
import { Output } from "./Output.tsx";

type Report = { kind: "run"; r: RunResult } | { kind: "test"; r: TestResult } | { kind: "error"; message: string };

export default function Exercise({ ex, lesson, n }: { ex: Ex; lesson: string; n: number }) {
  const key = `${lesson}/${ex.id}`;
  const progress = useProgress();
  const [code, setCode] = useState(() => savedCode(key) ?? ex.starter);
  const [busy, setBusy] = useState<"run" | "test" | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [hints, setHints] = useState(0);
  const [status] = usePyStatus();
  const passed = !!progress.passed[key];

  const edit = (v: string) => { setCode(v); saveCode(key, v === ex.starter ? null : v); };
  const go = async (kind: "run" | "test") => {
    if (busy) return;
    setBusy(kind);
    try {
      if (kind === "run") setReport({ kind, r: await runCode(code) });
      else {
        const r = await runTests(code, ex.tests);
        setReport({ kind, r });
        if (!r.error && r.results.length && r.results.every((t) => t.ok)) markPassed(key);
      }
    } catch (e) {
      setReport({ kind: "error", message: (e as Error).message });
    } finally {
      setBusy(null);
    }
  };
  const reveal = () => {
    if (!confirm("Show the solution? It replaces the code in the editor, so copy your attempt first if you want to keep it.")) return;
    markPeeked(key);
    edit(ex.solution);
    setReport(null);
  };

  const loading = status === "loading" || status === "idle";
  return (
    <section className={`exercise ${passed ? "passed" : ""}`} aria-labelledby={`ex-${ex.id}`} id={`ex-${ex.id}`}>
      <header className="ex-head">
        <span className="ex-n">Exercise {n}</span>
        <h3 id={`ex-${ex.id}`}>{ex.title}</h3>
        {passed && <span className="pass-tag">Passed</span>}
      </header>
      <Markdown text={ex.prompt} />
      <Editor value={code} onChange={edit} onSubmit={() => go("test")} label={`Code for exercise: ${ex.title}`}
              minLines={Math.min(18, Math.max(6, ex.starter.split("\n").length + 1))} />
      <div className="ex-bar">
        <button className="btn primary" onClick={() => go("test")} disabled={!!busy}>
          {busy === "test" ? "Checking…" : "Check"}<kbd>Ctrl ↵</kbd>
        </button>
        <button className="btn" onClick={() => go("run")} disabled={!!busy}>{busy === "run" ? "Running…" : "Run"}</button>
        {busy && <button className="btn ghost" onClick={stop}>Stop</button>}
        <span className="grow" />
        {ex.hints.length > 0 && hints < ex.hints.length && (
          <button className="btn ghost" onClick={() => setHints(hints + 1)}>{hints === 0 ? "Hint" : "Another hint"}</button>
        )}
        <button className="btn ghost" onClick={reveal}>Solution</button>
        <button className="btn ghost" onClick={() => { if (confirm("Put the starter code back?")) { edit(ex.starter); setReport(null); } }}>Reset</button>
      </div>
      {loading && busy && <p className="muted small">Starting Python in your browser. The first time takes a few seconds.</p>}
      {hints > 0 && (
        <ol className="hints">{ex.hints.slice(0, hints).map((h, i) => <li key={i}><Markdown text={h} inline /></li>)}</ol>
      )}
      {report && <Report report={report} />}
    </section>
  );
}

function Report({ report }: { report: Report }) {
  if (report.kind === "error") return <div className="report bad" role="status"><pre>{report.message}</pre></div>;
  if (report.kind === "run") return <div className="report" role="status"><Output r={report.r} /></div>;
  const { r } = report;
  const ok = r.results.filter((t) => t.ok).length;
  const all = !r.error && r.results.length > 0 && ok === r.results.length;
  return (
    <div className={`report ${all ? "good" : "bad"}`} role="status">
      <p className="verdict">{r.error ? "Your code stopped with an error before the tests ran." : all ? `All ${ok} checks passed.` : `${ok} of ${r.results.length} checks passed.`}</p>
      {r.error && <pre className="err">{r.error}</pre>}
      {r.results.length > 0 && (
        <ul className="tests">
          {r.results.map((t, i) => (
            <li key={i} className={t.ok ? "ok" : "no"}>
              <span className="mark" aria-hidden="true">{t.ok ? "✓" : "✗"}</span>
              <span className="sr">{t.ok ? "passed: " : "failed: "}</span>
              <span>{t.name}{!t.ok && t.message && <pre>{t.message}</pre>}</span>
            </li>
          ))}
        </ul>
      )}
      {r.stdout && <details><summary>Printed output</summary><pre>{r.stdout}</pre></details>}
    </div>
  );
}
