// The page's handle on the Python worker: one request at a time, a time limit, and a fresh
// worker after a stop or a time-out (the only way to interrupt running Python).
import { useSyncExternalStore } from "react";

export type PyStatus = "idle" | "loading" | "ready" | "running" | "failed";
export interface RunResult { stdout: string; error: string | null; figures: string[]; ms: number }
export interface TestResult { stdout: string; error: string | null; results: { name: string; ok: boolean; message?: string }[] }

const LIMIT_MS = 30_000;
let worker: Worker | null = null;
let status: PyStatus = "idle";
let failure = "";
let seq = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
const listeners = new Set<() => void>();
let ready: Promise<void> | null = null;

function setStatus(s: PyStatus, why = "") { status = s; failure = why; listeners.forEach((l) => l()); }

function spawn() {
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    if (e.data.ok) p.resolve(e.data.result);
    else p.reject(new Error(e.data.error));
  };
}

function send<T>(msg: Record<string, unknown>): Promise<T> {
  if (!worker) spawn();
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    worker!.postMessage({ ...msg, id, base: document.baseURI });
  });
}

/** Start Python in the background (about 10 MB the first time, then cached by the browser). */
export function warmUp(): Promise<void> {
  if (!ready) {
    setStatus("loading");
    ready = send<void>({ type: "init" }).then(() => setStatus("ready"), (e: Error) => {
      ready = null;
      worker?.terminate(); worker = null;
      setStatus("failed", e.message);
      throw e;
    });
  }
  return ready;
}

function restart(reason: string) {
  worker?.terminate();
  worker = null;
  ready = null;
  for (const [, p] of pending) p.reject(new Error(reason));
  pending.clear();
  warmUp().catch(() => {});
}

/** Stops whatever is running. Python cannot be interrupted, so the worker is replaced. */
export function stop() { restart("Stopped."); }

async function timed<T>(msg: Record<string, unknown>): Promise<T> {
  await warmUp();
  setStatus("running");
  let timer = 0;
  try {
    return await Promise.race([
      send<T>(msg),
      new Promise<T>((_, reject) => {
        timer = window.setTimeout(() => {
          reject(new Error(`Stopped after ${LIMIT_MS / 1000} seconds. Is there a loop that never ends?`));
          restart("time-out");
        }, LIMIT_MS);
      }),
    ]);
  } finally {
    clearTimeout(timer);
    if (status === "running") setStatus("ready");
  }
}

export const runCode = (code: string) => timed<RunResult>({ type: "run", code });
export const runTests = (code: string, tests: string) => timed<TestResult>({ type: "test", code, tests });

export function usePyStatus(): [PyStatus, string] {
  const s = useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => status);
  return [s, failure];
}
