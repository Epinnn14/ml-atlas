// All lessons, parsed once at start-up. Vite inlines the Markdown files at build time.
import { parseLesson, type Lesson, type Track, type Formula } from "./lib/lesson.ts";
import tracksJson from "../content/tracks.json";

const files = import.meta.glob("../content/*/*.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;

export const TRACKS: Track[] = tracksJson;
const trackOrder = new Map(TRACKS.map((t, i) => [t.id, i]));

export const LESSONS: Lesson[] = Object.entries(files)
  .map(([path, src]) => parseLesson(src, path))
  .sort((a, b) => (trackOrder.get(a.track)! - trackOrder.get(b.track)!) || a.order - b.order);

export const lessonById = new Map(LESSONS.map((l) => [l.id, l]));
export const lessonsOf = (track: string) => LESSONS.filter((l) => l.track === track);
export const FORMULAS: Formula[] = LESSONS.flatMap((l) => l.formulas);
export const exercisesOf = (l: Lesson) => l.blocks.filter((b) => b.type === "exercise");
export const TOTAL_EXERCISES = LESSONS.reduce((n, l) => n + exercisesOf(l).length, 0);

export function neighbours(id: string): [Lesson | undefined, Lesson | undefined] {
  const i = LESSONS.findIndex((l) => l.id === id);
  return [LESSONS[i - 1], LESSONS[i + 1]];
}
