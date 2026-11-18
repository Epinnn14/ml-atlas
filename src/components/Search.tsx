// Find a lesson or a formula by any word in its title, summary or section headings.
import { useMemo, useRef, useState } from "react";
import { FORMULAS, LESSONS, lessonById } from "../content.ts";

export default function Search() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLInputElement>(null);

  const hits = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const match = (s: string) => words.every((w) => s.toLowerCase().includes(w));
    const lessons = LESSONS.filter((l) => match([l.title, l.summary, ...l.headings.map((h) => h.text)].join(" ")))
      .map((l) => ({ href: `#/lesson/${l.id}`, title: l.title, sub: l.summary }));
    const formulas = FORMULAS.filter((f) => match(f.label))
      .map((f) => ({ href: `#/lesson/${f.lesson}`, title: f.label, sub: `Formula in ${lessonById.get(f.lesson)?.title}` }));
    return [...lessons, ...formulas].slice(0, 8);
  }, [q]);

  const go = (href: string) => { window.location.hash = href; setQ(""); setOpen(false); box.current?.blur(); };

  return (
    <div className="search" role="search">
      <input ref={box} type="search" placeholder="Search lessons and formulas" aria-label="Search lessons and formulas"
             value={q} aria-expanded={open && hits.length > 0} aria-controls="search-hits" aria-autocomplete="list" role="combobox"
             onChange={(e) => { setQ(e.target.value); setOpen(true); setActive(0); }}
             onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 150)}
             onKeyDown={(e) => {
               if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(hits.length - 1, a + 1)); }
               if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
               if (e.key === "Enter" && hits[active]) go(hits[active].href);
               if (e.key === "Escape") { setQ(""); setOpen(false); }
             }} />
      {open && q && (
        <ul id="search-hits" className="hits" role="listbox">
          {hits.length === 0 && <li className="none">Nothing matches “{q}”.</li>}
          {hits.map((h, i) => (
            <li key={h.href + h.title} role="option" aria-selected={i === active}>
              <a href={h.href} onMouseDown={(e) => { e.preventDefault(); go(h.href); }}>
                <strong>{h.title}</strong><span>{h.sub}</span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
