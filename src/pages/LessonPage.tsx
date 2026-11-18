import { useEffect } from "react";
import type { Lesson } from "../lib/lesson.ts";
import { lessonsOf, neighbours, TRACKS } from "../content.ts";
import { useProgress, visit } from "../lib/progress.ts";
import { lessonState } from "./Home.tsx";
import Markdown from "../components/Markdown.tsx";
import Exercise from "../components/Exercise.tsx";
import Quiz from "../components/Quiz.tsx";
import Level from "../components/Level.tsx";

const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

export default function LessonPage({ lesson }: { lesson: Lesson }) {
  const p = useProgress();
  const track = TRACKS.find((t) => t.id === lesson.track)!;
  const [prev, next] = neighbours(lesson.id);
  useEffect(() => { visit(lesson.id); document.title = `${lesson.title} · ML Atlas`; return () => { document.title = "ML Atlas"; }; }, [lesson.id, lesson.title]);

  let exN = 0;
  const exercises = lesson.blocks.filter((b) => b.type === "exercise");
  return (
    <div className="lesson-layout">
      <nav className="side" aria-label={`${track.title} lessons`}>
        <a className="side-track" href={`#/`}>{track.title}</a>
        <ol>
          {lessonsOf(track.id).map((l) => (
            <li key={l.id}>
              <a href={`#/lesson/${l.id}`} aria-current={l.id === lesson.id ? "page" : undefined} className={lessonState(l, p)}>
                {l.title}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <main id="main" tabIndex={-1} className="lesson">
        <header className="lesson-head">
          <p className="crumbs"><a href="#/">Curriculum</a> / {track.title}</p>
          <h1>{lesson.title}</h1>
          <p className="lede">{lesson.summary}</p>
          <p className="meta"><Level level={lesson.level} /> <span>{lesson.minutes} min</span> <span>{exercises.length} exercises</span></p>
        </header>
        {lesson.blocks.map((b, i) =>
          b.type === "md" ? <Markdown key={i} text={b.text} />
          : b.type === "exercise" ? <Exercise key={b.id} ex={b} lesson={lesson.id} n={++exN} />
          : <Quiz key={b.id} quiz={b} />)}
        <nav className="pager" aria-label="Next and previous lesson">
          {prev ? <a href={`#/lesson/${prev.id}`} className="prev"><span>Previous</span>{prev.title}</a> : <span />}
          {next ? <a href={`#/lesson/${next.id}`} className="next"><span>Next</span>{next.title}</a> : <span />}
        </nav>
      </main>

      <aside className="toc" aria-label="On this page">
        <p className="toc-h">On this page</p>
        <ul>
          {lesson.headings.map((h) => <li key={h.id}><button onClick={() => jump(h.id)}>{h.text}</button></li>)}
        </ul>
        {exercises.length > 0 && <>
          <p className="toc-h">Exercises</p>
          <ul>
            {exercises.map((e, i) => (
              <li key={e.id}>
                <button onClick={() => jump(`ex-${e.id}`)} className={p.passed[`${lesson.id}/${e.id}`] ? "done" : ""}>
                  {p.passed[`${lesson.id}/${e.id}`] ? "✓ " : `${i + 1}. `}{e.title}
                </button>
              </li>
            ))}
          </ul>
        </>}
      </aside>
    </div>
  );
}
