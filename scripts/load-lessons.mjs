// Reads and parses every lesson with the same parser the site uses.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseLesson } from "../src/lib/lesson.ts";

export function loadLessons(root = process.cwd()) {
  const content = join(root, "content");
  const tracks = JSON.parse(readFileSync(join(content, "tracks.json"), "utf8"));
  const lessons = [];
  for (const dir of readdirSync(content, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort()) {
    for (const f of readdirSync(join(content, dir)).filter((f) => f.endsWith(".md")).sort()) {
      const path = `content/${dir}/${f}`;
      lessons.push({ path, ...parseLesson(readFileSync(join(root, path), "utf8"), path) });
    }
  }
  return { tracks, lessons };
}
