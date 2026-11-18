// What the learner has done, kept in this browser only (localStorage).
import { useSyncExternalStore } from "react";

interface Progress {
  passed: Record<string, number>;      // "lesson/exercise" -> time first passed
  peeked: Record<string, number>;      // solution shown
  answers: Record<string, number>;     // "quiz-id/question" -> chosen option
  visited: Record<string, number>;     // lesson -> last visit
}
const KEY = "ml-atlas:progress";
const empty = (): Progress => ({ passed: {}, peeked: {}, answers: {}, visited: {} });

function load(): Progress {
  try { return { ...empty(), ...JSON.parse(localStorage.getItem(KEY) ?? "{}") }; } catch { return empty(); }
}
let state = load();
const listeners = new Set<() => void>();
function save(next: Progress) {
  state = next;
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* private mode: kept for this visit */ }
  listeners.forEach((l) => l());
}

export const useProgress = () => useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => state);
export const markPassed = (key: string) => { if (!state.passed[key]) save({ ...state, passed: { ...state.passed, [key]: Date.now() } }); };
export const markPeeked = (key: string) => save({ ...state, peeked: { ...state.peeked, [key]: Date.now() } });
export const answer = (key: string, choice: number) => save({ ...state, answers: { ...state.answers, [key]: choice } });
export const visit = (lesson: string) => save({ ...state, visited: { ...state.visited, [lesson]: Date.now() } });
export const resetProgress = () => save(empty());

/** Code the learner typed, per exercise, so a reload keeps it. */
export const savedCode = (key: string): string | null => { try { return localStorage.getItem("ml-atlas:code:" + key); } catch { return null; } };
export const saveCode = (key: string, code: string | null) => {
  try { if (code === null) localStorage.removeItem("ml-atlas:code:" + key); else localStorage.setItem("ml-atlas:code:" + key, code); } catch { /* ignore */ }
};
