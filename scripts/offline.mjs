// Makes the site work with no internet: downloads the full Pyodide release once and keeps only
// the wheels the lessons use (numpy, matplotlib and their dependencies) in public/pyodide.
//   npm run offline                       download from GitHub (about 400 MB, kept in .cache/)
//   PYODIDE_TARBALL=path npm run offline  use a tarball you already have
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync, rmSync, readdirSync, renameSync } from "node:fs";
import { join, dirname } from "node:path";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const require = createRequire(import.meta.url);
const version = require("pyodide/package.json").version;
const root = process.cwd();
const out = join(root, "public", "pyodide");
const cache = join(root, ".cache");
const WANT = ["numpy", "matplotlib"];

const lock = JSON.parse(readFileSync(join(dirname(require.resolve("pyodide/package.json")), "pyodide-lock.json"), "utf8"));
const need = new Set();
const walk = (n) => { if (need.has(n)) return; need.add(n); for (const d of lock.packages[n].depends ?? []) walk(d); };
WANT.forEach(walk);
const files = [...need].map((n) => lock.packages[n].file_name);

let tarball = process.env.PYODIDE_TARBALL;
if (!tarball) {
  mkdirSync(cache, { recursive: true });
  tarball = join(cache, `pyodide-${version}.tar.bz2`);
  if (!existsSync(tarball)) {
    const url = `https://github.com/pyodide/pyodide/releases/download/${version}/pyodide-${version}.tar.bz2`;
    console.log(`downloading ${url} (about 400 MB, once)`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download failed: ${res.status}`);
    await pipeline(Readable.fromWeb(res.body), createWriteStream(tarball + ".part"));
    renameSync(tarball + ".part", tarball);
  }
}
const tmp = join(cache, "extract");
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp, { recursive: true });
console.log(`extracting ${files.length} wheels: ${[...need].sort().join(", ")}`);
execFileSync("tar", ["-xjf", tarball, "-C", tmp, ...files.map((f) => `pyodide/${f}`)], { stdio: "inherit" });
mkdirSync(out, { recursive: true });
for (const f of readdirSync(join(tmp, "pyodide"))) renameSync(join(tmp, "pyodide", f), join(out, f));
rmSync(tmp, { recursive: true, force: true });
writeFileSync(join(out, "offline.json"), JSON.stringify({ version, packages: [...need].sort() }, null, 2));
console.log("done: lessons now run with no internet connection");
