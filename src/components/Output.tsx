import type { RunResult } from "../py/runner.ts";

export function Output({ r }: { r: RunResult }) {
  const empty = !r.stdout && !r.error && r.figures.length === 0;
  return (
    <div className="output">
      {r.stdout && <pre>{r.stdout}</pre>}
      {r.error && <pre className="err">{r.error}</pre>}
      {r.figures.map((f, i) => <img key={i} src={`data:image/png;base64,${f}`} alt={`Figure ${i + 1} drawn by the code`} />)}
      {empty && <p className="muted small">The code ran in {r.ms} ms and printed nothing. Use print() to see values.</p>}
    </div>
  );
}
