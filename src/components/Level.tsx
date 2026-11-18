import type { Level as L } from "../lib/lesson.ts";
import { LEVELS } from "../lib/lesson.ts";

const NAME: Record<L, string> = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced", senior: "Senior" };

/** Difficulty as one to four filled steps plus the word, so it never relies on colour. */
export default function Level({ level }: { level: L }) {
  const n = LEVELS.indexOf(level) + 1;
  return (
    <span className="level" title={`${NAME[level]} (${n} of 4)`}>
      <span className="steps" aria-hidden="true">{LEVELS.map((_, i) => <i key={i} className={i < n ? "on" : ""} />)}</span>
      {NAME[level]}
    </span>
  );
}
