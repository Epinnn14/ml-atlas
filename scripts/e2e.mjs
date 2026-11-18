// Browser check of the built site: `npm run build && npm run e2e`.
// Opens every page on a phone and a desktop, in light and dark, and fails on console errors,
// sideways scrolling or missing content. In a few lessons it shows the solution, runs the tests
// in the browser's Python and expects them all to pass; one is typed in by hand.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { loadLessons } from "./load-lessons.mjs";

const out = process.argv[2] ?? ".cache/shots";
mkdirSync(out, { recursive: true });
const { lessons } = loadLessons();
const port = 4317;
const server = spawn("npx", ["vite", "preview", "--port", String(port), "--strictPort"], { stdio: "ignore" });
const base = `http://localhost:${port}/`;
for (let i = 0; i < 50; i++) { try { await fetch(base); break; } catch { await new Promise((r) => setTimeout(r, 200)); } }

const problems = [];
const browser = await chromium.launch();
async function open(label, opts, theme) {
  const ctx = await browser.newContext(opts);
  await ctx.addInitScript((t) => { try { localStorage.setItem("ml-atlas:theme", t); } catch {} }, theme);
  const page = await ctx.newPage();
  page.on("pageerror", (e) => problems.push(`${label}: page error ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`${label}: console ${m.text()}`); });
  page.on("dialog", (d) => d.accept());
  return page;
}
async function overflow(page, label, where) {
  const px = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (px > 1) problems.push(`${label}: ${where} scrolls sideways by ${px}px`);
}

// 1. every lesson renders, on a phone (light) and a desktop (dark)
for (const [label, opts, theme] of [["phone", { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }, "light"],
                                     ["desktop", { viewport: { width: 1400, height: 900 } }, "dark"]]) {
  const page = await open(label, opts, theme);
  await page.goto(base);
  await page.getByRole("heading", { level: 1 }).first().waitFor();
  await overflow(page, label, "home");
  await page.screenshot({ path: `${out}/${label}-home.png`, fullPage: label === "desktop" });
  for (const l of lessons) {
    await page.goto(`${base}#/lesson/${l.id}`);
    try { await page.getByRole("heading", { level: 1, name: l.title }).waitFor({ timeout: 10000 }); }
    catch { problems.push(`${label}: lesson ${l.id} did not render`); continue; }
    const katexErrors = await page.locator(".katex-error").count();
    if (katexErrors) problems.push(`${label}: ${l.id} has ${katexErrors} formulas KaTeX could not render`);
    await overflow(page, label, l.id);
  }
  await page.screenshot({ path: `${out}/${label}-lesson-last.png` });
  for (const p of ["formulas", "playground"]) {
    await page.goto(`${base}#/${p}`);
    await page.getByRole("heading", { level: 1 }).first().waitFor();
    await overflow(page, label, p);
    await page.screenshot({ path: `${out}/${label}-${p}.png`, fullPage: label === "desktop" });
  }
  await page.context().close();
}

// 2. Python in the browser: solutions pass, typed code passes, the playground draws a figure
{
  const page = await open("python", { viewport: { width: 1400, height: 900 } }, "light");
  const pick = lessons.filter((l, i) => i % Math.max(1, Math.floor(lessons.length / 6)) === 0).slice(0, 6);
  for (const l of pick) {
    await page.goto(`${base}#/lesson/${l.id}`);
    const ex = page.locator("section.exercise").first();
    await ex.getByRole("button", { name: "Solution" }).click();
    await ex.getByRole("button", { name: /^Check/ }).click();
    try { await ex.getByText(/All \d+ checks passed\./).waitFor({ timeout: 90000 }); }
    catch { problems.push(`python: solution of the first exercise in ${l.id} did not pass in the browser: ${await ex.locator(".report").innerText().catch(() => "no report")}`); }
  }
  // typed by hand, then the first lesson shows as passed on the home page
  const first = lessons[0];
  const ex0 = first.blocks.find((b) => b.type === "exercise");
  await page.goto(`${base}#/lesson/${first.id}`);
  const ex = page.locator("section.exercise").first();
  await ex.getByRole("button", { name: "Reset" }).click();
  await ex.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText(ex0.solution);
  await page.keyboard.press("ControlOrMeta+Enter");
  try { await ex.getByText(/All \d+ checks passed\./).waitFor({ timeout: 60000 }); }
  catch { problems.push(`python: typed solution did not pass: ${await ex.locator(".report").innerText().catch(() => "no report")}`); }
  await page.screenshot({ path: `${out}/exercise-passed.png`, fullPage: false });
  // a wrong answer shows which checks failed
  await ex.getByRole("button", { name: "Reset" }).click();
  await ex.getByRole("button", { name: /^Check/ }).click();
  await ex.locator(".report.bad").waitFor({ timeout: 30000 }).catch(() => problems.push("python: starter code did not show a failing report"));
  // quiz: answer the first question
  const q = page.locator("fieldset.question").first();
  await q.locator("label.option").first().click();
  await q.locator(".why").waitFor({ timeout: 5000 }).catch(() => problems.push("quiz: no explanation after answering"));
  // playground draws a figure
  await page.goto(`${base}#/playground`);
  await page.getByRole("button", { name: /^Run/ }).click();
  await page.locator(".output img").waitFor({ timeout: 90000 }).catch(() => problems.push("playground: no figure"));
  await page.screenshot({ path: `${out}/playground-run.png` });
  // an endless loop is stopped
  await page.locator(".cm-content").click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.insertText("while True:\n    pass\n");
  await page.getByRole("button", { name: /^Run/ }).click();
  await page.getByRole("button", { name: "Stop" }).click();
  await page.getByRole("button", { name: /^Run/ }).click();
  await page.getByText(/printed nothing/).waitFor({ timeout: 30000 }).catch(() => {}); // stopped run, then the loop again
  await page.getByRole("button", { name: "Stop" }).click().catch(() => {});
  await page.context().close();
}

await browser.close();
server.kill();
if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
console.log(`e2e ok: ${lessons.length} lessons on phone and desktop, Python checks passed`);
