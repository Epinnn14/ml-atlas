// Puts the built site and the start scripts into ml-atlas-site.zip for the GitHub release.
import { cpSync, existsSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";

if (!existsSync("dist/pyodide/offline.json")) throw new Error("run `npm run offline` before building, so the zip works without internet");
cpSync("site-extras", "dist", { recursive: true });
rmSync("ml-atlas-site", { recursive: true, force: true });
cpSync("dist", "ml-atlas-site", { recursive: true });
rmSync("ml-atlas-site.zip", { force: true });
execFileSync("zip", ["-qr", "ml-atlas-site.zip", "ml-atlas-site"]);
rmSync("ml-atlas-site", { recursive: true, force: true });
console.log("ml-atlas-site.zip ready");
