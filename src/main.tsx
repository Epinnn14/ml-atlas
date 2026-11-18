import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/source-serif-4/wght.css";
import "@fontsource-variable/source-sans-3/wght.css";
import "@fontsource-variable/source-code-pro/wght.css";
import "katex/dist/katex.min.css";
import "./styles.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
