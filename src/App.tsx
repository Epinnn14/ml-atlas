import { useEffect, useState } from "react";
import { usePyStatus, warmUp } from "./py/runner.ts";
import Home from "./pages/Home.tsx";
import LessonPage from "./pages/LessonPage.tsx";
import Formulas from "./pages/Formulas.tsx";
import Playground from "./pages/Playground.tsx";
import Search from "./components/Search.tsx";
import { lessonById } from "./content.ts";

function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [r, setR] = useState(read);
  useEffect(() => {
    const on = () => { setR(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}

type Theme = "light" | "dark";
function useTheme(): [Theme, () => void] {
  const sys = () => (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const [t, setT] = useState<Theme>(() => (document.documentElement.dataset.theme as Theme) || sys());
  useEffect(() => {
    document.documentElement.dataset.theme = t;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", t === "dark" ? "#15171b" : "#f6f4ee");
  }, [t]);
  return [t, () => {
    const next = t === "dark" ? "light" : "dark";
    try { localStorage.setItem("ml-atlas:theme", next); } catch { /* ignore */ }
    setT(next);
  }];
}

function PyBadge() {
  const [s, why] = usePyStatus();
  const text = { idle: "Python not started", loading: "Starting Python…", ready: "Python ready", running: "Running…", failed: "Python failed to start" }[s];
  return (
    <span className={`py-badge ${s}`} title={s === "failed" ? why : "Python runs inside your browser, nothing is sent anywhere."}>
      <span className="dot" aria-hidden="true" />{text}
    </span>
  );
}

export default function App() {
  const route = useRoute();
  const [theme, toggle] = useTheme();
  const page = route[0] ?? "";
  // start Python as soon as a page with code is open
  useEffect(() => { if (page === "lesson" || page === "playground") warmUp().catch(() => {}); }, [page]);

  let body;
  if (page === "lesson" && route[1] && lessonById.has(route[1])) body = <LessonPage key={route[1]} lesson={lessonById.get(route[1])!} />;
  else if (page === "formulas") body = <Formulas />;
  else if (page === "playground") body = <Playground />;
  else if (page === "" ) body = <Home />;
  else body = (
    <main id="main" className="page narrow">
      <h1>Page not found</h1>
      <p>There is nothing at this address. <a href="#/">Go to the curriculum</a>.</p>
    </main>
  );

  return (
    <>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>Skip to content</a>
      <header className="top">
        <a className="brand" href="#/" aria-label="ML Atlas, home">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true"><rect width="32" height="32" rx="6" fill="var(--accent)" /><path d="M6 22c4-1 6-9 10-9s6 6 10 5" stroke="var(--on-accent)" strokeWidth="2.4" fill="none" strokeLinecap="round" /><circle cx="16" cy="13" r="2.4" fill="var(--on-accent)" /></svg>
          <span>ML Atlas</span>
        </a>
        <nav aria-label="Main">
          <a href="#/" aria-current={page === "" || page === "lesson" ? "page" : undefined}>Curriculum</a>
          <a href="#/formulas" aria-current={page === "formulas" ? "page" : undefined}>Formulas</a>
          <a href="#/playground" aria-current={page === "playground" ? "page" : undefined}>Playground</a>
        </nav>
        <Search />
        <PyBadge />
        <button className="icon-btn" onClick={toggle} aria-label={theme === "dark" ? "Use light theme" : "Use dark theme"} title={theme === "dark" ? "Light theme" : "Dark theme"}>
          {theme === "dark"
            ? <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
            : <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /></svg>}
        </button>
      </header>
      {body}
    </>
  );
}
