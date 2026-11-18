// Python in a Web Worker, so a slow or endless loop never freezes the page.
// The runtime is loaded from ./pyodide/ (copied there by the build); packages come from the same
// folder when `npm run offline` was run, otherwise once from the Pyodide CDN.
import harness from "./harness.py?raw";

type Req = { id: number; type: "init" } | { id: number; type: "run"; code: string } | { id: number; type: "test"; code: string; tests: string };

const CDN = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null;
let starting: Promise<void> | null = null;

async function start(base: string) {
  const index = new URL("pyodide/", base).href;
  const offline = await fetch(index + "offline.json").then((r) => r.ok).catch(() => false);
  const { loadPyodide } = await import(/* @vite-ignore */ index + "pyodide.mjs");
  py = await loadPyodide({ indexURL: index, packageBaseUrl: offline ? index : CDN });
  await py.loadPackage(["numpy"], { messageCallback: () => {} });
  py.runPython(harness);
}

self.onmessage = async (e: MessageEvent<Req & { base: string }>) => {
  const msg = e.data;
  try {
    if (!starting) starting = start(msg.base);
    await starting;
    if (msg.type === "init") return postMessage({ id: msg.id, ok: true });
    const code = msg.type === "test" ? msg.code + "\n" + msg.tests : msg.code;
    await py.loadPackagesFromImports(code, { messageCallback: () => {} });
    const fn = py.globals.get(msg.type === "test" ? "run_tests" : "run_code");
    const out = msg.type === "test" ? fn(msg.code, msg.tests) : fn(msg.code);
    fn.destroy();
    postMessage({ id: msg.id, ok: true, result: JSON.parse(out) });
  } catch (err) {
    starting = msg.type === "init" ? null : starting;
    postMessage({ id: msg.id, ok: false, error: String((err as Error)?.message ?? err) });
  }
};
