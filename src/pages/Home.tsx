// The curriculum: every track and lesson, with what this learner has finished.
import { exercisesOf, LESSONS, lessonsOf, TOTAL_EXERCISES, TRACKS } from "../content.ts";
import type { Lesson } from "../lib/lesson.ts";
import { useProgress } from "../lib/progress.ts";
import Level from "../components/Level.tsx";

export function lessonState(l: Lesson, p: ReturnType<typeof useProgress>): "done" | "started" | "new" {
  const ex = exercisesOf(l);
  if (ex.length && ex.every((e) => p.passed[`${l.id}/${e.id}`])) return "done";
  return p.visited[l.id] ? "started" : "new";
}

export default function Home() {
  const p = useProgress();
  const passed = Object.keys(p.passed).length;
  const last = Object.entries(p.visited).sort((a, b) => b[1] - a[1])[0]?.[0];
  const resume = LESSONS.find((l) => l.id === last);
  const minutes = LESSONS.reduce((n, l) => n + l.minutes, 0);

  return (
    <main id="main" tabIndex={-1} className="page home">
      <section className="intro">
        <h1>Machine learning, worked out by hand.</h1>
        <p className="lede">
          {LESSONS.length} lessons from vectors and gradients to Transformers, diffusion and reinforcement learning.
          Each one explains the idea, derives the formulas, and asks you to implement it in Python with NumPy.
          Your code runs and is tested right here in the browser, on your own computer.
        </p>
        <div className="intro-row">
          {resume
            ? <a className="btn primary big" href={`#/lesson/${resume.id}`}>Continue: {resume.title}</a>
            : <a className="btn primary big" href={`#/lesson/${LESSONS[0].id}`}>Start with {LESSONS[0].title}</a>}
          <p className="muted">
            {passed} of {TOTAL_EXERCISES} exercises passed · about {Math.round(minutes / 60)} hours in total
          </p>
        </div>
      </section>

      <ol className="tracks">
        {TRACKS.map((t, ti) => {
          const ls = lessonsOf(t.id);
          const done = ls.filter((l) => lessonState(l, p) === "done").length;
          return (
            <li key={t.id} className="track" id={`track-${t.id}`}>
              <div className="track-head">
                <span className="track-n">Part {ti + 1}</span>
                <h2>{t.title}</h2>
                <p>{t.blurb}</p>
                <p className="small muted">{done} of {ls.length} lessons finished</p>
              </div>
              <ol className="lesson-list">
                {ls.map((l) => {
                  const st = lessonState(l, p);
                  const ex = exercisesOf(l).length;
                  return (
                    <li key={l.id} className={st}>
                      <a href={`#/lesson/${l.id}`}>
                        <span className="state" aria-label={st === "done" ? "finished" : st === "started" ? "started" : "not started"}>
                          {st === "done" ? "✓" : ""}
                        </span>
                        <span className="l-main">
                          <strong>{l.title}</strong>
                          <span className="l-sum">{l.summary}</span>
                        </span>
                        <span className="l-meta">
                          <Level level={l.level} />
                          <span>{l.minutes} min · {ex} exercise{ex === 1 ? "" : "s"}</span>
                        </span>
                      </a>
                    </li>
                  );
                })}
              </ol>
            </li>
          );
        })}
      </ol>
      <footer className="foot small muted">
        Progress and your code are saved in this browser only. Python runs locally with Pyodide; nothing you type leaves your computer.
        {" "}<a href="https://github.com/Epinnn14/ml-atlas" target="_blank" rel="noopener">Source on GitHub</a>
      </footer>
    </main>
  );
}
