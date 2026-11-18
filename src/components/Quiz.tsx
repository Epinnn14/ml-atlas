// Check-your-understanding questions: pick an answer, see at once why it is right or wrong.
import type { Quiz as Q } from "../lib/lesson.ts";
import { answer, useProgress } from "../lib/progress.ts";
import Markdown from "./Markdown.tsx";

export default function Quiz({ quiz }: { quiz: Q }) {
  const progress = useProgress();
  const right = quiz.questions.filter((q, i) => q.options[progress.answers[`${quiz.id}/${i}`]]?.correct).length;
  const answered = quiz.questions.filter((_, i) => progress.answers[`${quiz.id}/${i}`] !== undefined).length;
  return (
    <section className="quiz" aria-label="Check your understanding">
      <header className="ex-head">
        <span className="ex-n">Check your understanding</span>
        {answered > 0 && <span className="muted small">{right} of {quiz.questions.length} right</span>}
      </header>
      {quiz.questions.map((q, i) => {
        const key = `${quiz.id}/${i}`;
        const chosen = progress.answers[key];
        const done = chosen !== undefined;
        return (
          <fieldset key={i} className="question">
            <legend><span className="qn">{i + 1}.</span> <Markdown text={q.q} inline /></legend>
            {q.options.map((o, j) => {
              const cls = !done ? "" : o.correct ? "right" : j === chosen ? "wrong" : "";
              return (
                <label key={j} className={`option ${cls}`}>
                  <input type="radio" name={key} checked={chosen === j} onChange={() => answer(key, j)} />
                  <Markdown text={o.text} inline />
                  {done && o.correct && <span className="tag">correct answer</span>}
                  {done && j === chosen && !o.correct && <span className="tag">your answer</span>}
                </label>
              );
            })}
            {done && (
              <div className={`why ${q.options[chosen]?.correct ? "good" : "bad"}`} role="status">
                <strong>{q.options[chosen]?.correct ? "Right." : "Not quite."}</strong> <Markdown text={q.why} inline />
              </div>
            )}
          </fieldset>
        );
      })}
    </section>
  );
}
