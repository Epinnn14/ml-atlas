// Copies the Pyodide runtime from node_modules into public/pyodide so the site never needs a CDN
// for Python itself. Packages (numpy, matplotlib) come from public/pyodide too after
// `npm run offline`; otherwise the browser fetches them once from the Pyodide CDN.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const src = dirname(require.resolve("pyodide/package.json"));
const out = join(process.cwd(), "public", "pyodide");
mkdirSync(out, { recursive: true });
for (const f of ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) {
  copyFileSync(join(src, f), join(out, f));
}
const offline = existsSync(join(out, "offline.json"));
console.log(`pyodide ${require("pyodide/package.json").version} copied to public/pyodide (${offline ? "packages local, works offline" : "packages from CDN; run `npm run offline` to keep them local"})`);
