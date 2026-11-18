// Every named formula from the lessons on one page, grouped by part, searchable.
import { useMemo, useState } from "react";
import { FORMULAS, lessonById, TRACKS } from "../content.ts";
import { tex } from "../lib/markdown.ts";

export default function Formulas() {
  const [q, setQ] = useState("");
  const shown = useMemo(() => {
    const w = q.toLowerCase().trim();
    return w ? FORMULAS.filter((f) => (f.label + " " + lessonById.get(f.lesson)!.title).toLowerCase().includes(w)) : FORMULAS;
  }, [q]);
  return (
    <main id="main" tabIndex={-1} className="page">
      <header className="page-head">
        <h1>Formula sheet</h1>
        <p className="lede">All {FORMULAS.length} formulas from the lessons. Each links back to where it is derived and used.</p>
        <input type="search" className="filter" placeholder="Filter, for example: loss, gradient, attention" aria-label="Filter formulas"
               value={q} onChange={(e) => setQ(e.target.value)} />
      </header>
      {shown.length === 0 && <p className="muted">No formula matches “{q}”. Try a shorter word.</p>}
      {TRACKS.map((t) => {
        const fs = shown.filter((f) => lessonById.get(f.lesson)!.track === t.id);
        if (!fs.length) return null;
        return (
          <section key={t.id} className="f-group">
            <h2>{t.title}</h2>
            <dl className="f-list">
              {fs.map((f) => (
                <div key={f.lesson + f.label} className="f-item">
                  <dt>{f.label}<a href={`#/lesson/${f.lesson}`}>{lessonById.get(f.lesson)!.title}</a></dt>
                  <dd dangerouslySetInnerHTML={{ __html: tex(f.tex, true) }} />
                </div>
              ))}
            </dl>
          </section>
        );
      })}
    </main>
  );
}
