// Runs every exercise in Pyodide (the same Python the browser uses): the solution must pass
// all of its tests, and the untouched starter must fail at least one. Needs `npm run offline`.
//   npm run check               all lessons
//   npm run check -- pca knn    only lessons whose id contains one of the words
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadPyodide } from "pyodide";
import { loadLessons } from "./load-lessons.mjs";

const root = process.cwd();
const pkg = join(root, "public", "pyodide") + "/";
if (!existsSync(pkg + "offline.json")) { console.error("Run `npm run offline` first so the packages are local."); process.exit(1); }
const only = process.argv.slice(2);
const { lessons } = loadLessons(root);
const py = await loadPyodide({ packageBaseUrl: pkg });
await py.loadPackage(["numpy", "matplotlib"], { messageCallback: () => {} });
py.runPython(readFileSync(join(root, "src", "py", "harness.py"), "utf8"));
const runTests = py.globals.get("run_tests");

let fails = 0, count = 0;
const t0 = Date.now();
for (const l of lessons) {
  if (only.length && !only.some((w) => l.id.includes(w))) continue;
  for (const b of l.blocks) {
    if (b.type !== "exercise") continue;
    count++;
    const t = Date.now();
    const sol = JSON.parse(runTests(b.solution, b.tests));
    const okSol = !sol.error && sol.results.length > 0 && sol.results.every((r) => r.ok);
    const st = JSON.parse(runTests(b.starter, b.tests));
    const okStarter = !!st.error || st.results.some((r) => !r.ok);
    const ms = Date.now() - t;
    if (!okSol || !okStarter) {
      fails++;
      console.error(`FAIL ${l.id}/${b.id}`);
      if (!okSol) console.error("  solution:", sol.error ?? sol.results.filter((r) => !r.ok).map((r) => `${r.name}: ${r.message}`).join("\n  "));
      if (!okStarter) console.error("  the starter already passes every test");
    } else if (ms > 8000) console.warn(`slow ${l.id}/${b.id}: ${ms} ms`);
  }
}
console.log(`${count - fails} of ${count} exercises ok in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
process.exit(fails ? 1 : 0);
