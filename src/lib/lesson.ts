// Lessons are Markdown files with a small front matter and three kinds of embedded blocks:
// formulas (collected into the formula sheet), exercises (starter, solution, tests, hints) and
// quizzes. The same parser runs in the browser and in the Node checks.

export type Level = "beginner" | "intermediate" | "advanced" | "senior";
export const LEVELS: Level[] = ["beginner", "intermediate", "advanced", "senior"];

export interface Exercise {
  type: "exercise";
  id: string;
  title: string;
  prompt: string;
  starter: string;
  solution: string;
  tests: string;
  hints: string[];
}
export interface QuizQuestion { q: string; options: { text: string; correct: boolean }[]; why: string }
export interface Quiz { type: "quiz"; id: string; questions: QuizQuestion[] }
export interface Prose { type: "md"; text: string }
export type Block = Prose | Exercise | Quiz;

export interface Formula { label: string; tex: string; lesson: string }
export interface Lesson {
  id: string;
  track: string;
  order: number;
  title: string;
  level: Level;
  minutes: number;
  summary: string;
  blocks: Block[];
  formulas: Formula[];
  headings: { id: string; text: string }[];
}

export const slugify = (s: string) =>
  s.toLowerCase().replace(/[`$\\{}^_]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

function frontMatter(src: string, file: string): [Record<string, string>, string] {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n/.exec(src);
  if (!m) throw new Error(`${file}: missing front matter`);
  const meta: Record<string, string> = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = /^(\w+):\s*(.*)$/.exec(line);
    if (kv) meta[kv[1]] = kv[2].trim();
  }
  return [meta, src.slice(m[0].length)];
}

function codeBlocks(body: string): { kind: string; code: string }[] {
  const out: { kind: string; code: string }[] = [];
  const re = /^~~~python (\w+)\r?\n([\s\S]*?)^~~~\s*$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) out.push({ kind: m[1], code: m[2].replace(/\s+$/, "") + "\n" });
  return out;
}

function parseExercise(id: string, title: string, body: string, file: string): Exercise {
  const blocks = codeBlocks(body);
  const get = (k: string) => {
    const b = blocks.find((x) => x.kind === k);
    if (!b) throw new Error(`${file}: exercise ${id} has no ${k}`);
    return b.code;
  };
  const hints = [...body.matchAll(/<!--\s*hint:\s*([\s\S]*?)\s*-->/g)].map((h) => h[1]);
  const prompt = body
    .replace(/^~~~python \w+\r?\n[\s\S]*?^~~~\s*$/gm, "")
    .replace(/<!--\s*hint:[\s\S]*?-->/g, "")
    .trim();
  return { type: "exercise", id, title, prompt, starter: get("starter"), solution: get("solution"), tests: get("tests"), hints };
}

function parseQuiz(id: string, body: string, file: string): Quiz {
  const questions: QuizQuestion[] = [];
  let cur: QuizQuestion | null = null;
  for (const raw of body.split(/\r?\n/)) {
    const line = raw.trimEnd();
    let m: RegExpExecArray | null;
    if ((m = /^\d+\.\s+(.+)$/.exec(line))) {
      cur = { q: m[1], options: [], why: "" };
      questions.push(cur);
    } else if ((m = /^\s+- \[( |x)\]\s+(.+)$/.exec(line)) && cur) {
      cur.options.push({ text: m[2], correct: m[1] === "x" });
    } else if ((m = /^\s+>\s?(.*)$/.exec(line)) && cur) {
      cur.why += (cur.why ? " " : "") + m[1];
    } else if (line.trim() && cur) {
      cur.q += " " + line.trim();
    }
  }
  for (const q of questions) {
    if (q.options.filter((o) => o.correct).length !== 1) throw new Error(`${file}: quiz question needs exactly one [x]: ${q.q}`);
    if (!q.why) throw new Error(`${file}: quiz question needs an explanation (> ...): ${q.q}`);
  }
  if (!questions.length) throw new Error(`${file}: empty quiz`);
  return { type: "quiz", id, questions };
}

/** `path` like content/02-classical/03-logistic-regression.md */
export function parseLesson(src: string, path: string): Lesson {
  const parts = path.replace(/\\/g, "/").split("/");
  const file = parts[parts.length - 1];
  const track = parts[parts.length - 2].replace(/^\d+-/, "");
  const fm = /^(\d+)-(.+)\.md$/.exec(file);
  if (!fm) throw new Error(`${path}: name lessons NN-slug.md`);
  const [meta, body] = frontMatter(src, path);
  for (const k of ["title", "level", "minutes", "summary"]) if (!meta[k]) throw new Error(`${path}: front matter needs ${k}`);
  if (!LEVELS.includes(meta.level as Level)) throw new Error(`${path}: level must be one of ${LEVELS.join(", ")}`);
  const id = fm[2];

  const blocks: Block[] = [];
  const re = /<!--\s*exercise:\s*([\w-]+)\s*\|\s*(.+?)\s*-->([\s\S]*?)<!--\s*\/exercise\s*-->|<!--\s*quiz\s*-->([\s\S]*?)<!--\s*\/quiz\s*-->/g;
  let last = 0;
  let quizN = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) {
    const before = body.slice(last, m.index).trim();
    if (before) blocks.push({ type: "md", text: before });
    if (m[1]) blocks.push(parseExercise(m[1], m[2], m[3], path));
    else blocks.push(parseQuiz(`${id}-quiz-${++quizN}`, m[4], path));
    last = m.index + m[0].length;
  }
  const rest = body.slice(last).trim();
  if (rest) blocks.push({ type: "md", text: rest });

  const formulas: Formula[] = [];
  for (const f of body.matchAll(/<!--\s*formula:\s*(.+?)\s*-->\s*\r?\n\s*\$\$([\s\S]+?)\$\$/g)) {
    formulas.push({ label: f[1], tex: f[2].trim(), lesson: id });
  }
  const headings = [...body.matchAll(/^## (.+)$/gm)].map((h) => ({ id: slugify(h[1]), text: h[1] }));

  return {
    id, track, order: Number(fm[1]), title: meta.title, level: meta.level as Level, minutes: Number(meta.minutes),
    summary: meta.summary, blocks, formulas, headings,
  };
}

export interface Track { id: string; title: string; blurb: string }
