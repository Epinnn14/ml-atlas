// Fails the build on broken lessons: bad front matter, unknown track, duplicate ids,
// TeX that KaTeX cannot render, links to lessons that do not exist.
import katex from "katex";
import { loadLessons } from "./load-lessons.mjs";

const { tracks, lessons } = loadLessons();
const problems = [];
const trackIds = new Set(tracks.map((t) => t.id));
const ids = new Set();
for (const l of lessons) {
  if (!trackIds.has(l.track)) problems.push(`${l.path}: folder is not a track in tracks.json`);
  if (ids.has(l.id)) problems.push(`${l.path}: lesson id "${l.id}" is used twice`);
  ids.add(l.id);
  const exIds = new Set();
  for (const b of l.blocks) if (b.type === "exercise") {
    if (exIds.has(b.id)) problems.push(`${l.path}: exercise id "${b.id}" is used twice`);
    exIds.add(b.id);
    if (!/def test_/.test(b.tests)) problems.push(`${l.path}: exercise ${b.id} has no test_ functions`);
  }
  if (!l.blocks.some((b) => b.type === "exercise")) problems.push(`${l.path}: no exercise`);
  if (!l.blocks.some((b) => b.type === "quiz")) problems.push(`${l.path}: no quiz`);
  const text = l.blocks.map((b) => b.type === "md" ? b.text : b.type === "exercise" ? b.prompt + " " + b.hints.join(" ") : b.questions.map((q) => [q.q, q.why, ...q.options.map((o) => o.text)].join(" ")).join(" ")).join("\n")
    .replace(/^~~~[\s\S]*?^~~~$/gm, "").replace(/^```[\s\S]*?^```$/gm, "").replace(/`[^`\n]*`/g, "");
  for (const m of text.matchAll(/\$\$([\s\S]+?)\$\$|(?<!\\)\$((?:\\\$|[^$\n])+?)\$/g)) {
    const src = m[1] ?? m[2];
    try { katex.renderToString(src, { displayMode: !!m[1], throwOnError: true, strict: "ignore" }); }
    catch (e) { problems.push(`${l.path}: TeX error in "${src.slice(0, 60)}": ${e.message.split("\n")[0]}`); }
  }
  for (const m of text.matchAll(/\]\(#\/lesson\/([\w-]+)\)/g)) if (!lessons.some((x) => x.id === m[1])) problems.push(`${l.path}: link to unknown lesson ${m[1]}`);
}
if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
const ex = lessons.reduce((n, l) => n + l.blocks.filter((b) => b.type === "exercise").length, 0);
const qs = lessons.reduce((n, l) => n + l.blocks.filter((b) => b.type === "quiz").reduce((k, q) => k + q.questions.length, 0), 0);
console.log(`content ok: ${lessons.length} lessons, ${ex} exercises, ${qs} quiz questions, ${lessons.reduce((n, l) => n + l.formulas.length, 0)} formulas`);
