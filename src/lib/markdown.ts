// Markdown with KaTeX math ($inline$, $$display$$), heading anchors and Python highlighting.
import { Marked, type Tokens } from "marked";
import katex from "katex";
import { highlightCode, classHighlighter } from "@lezer/highlight";
import { pythonLanguage } from "@codemirror/lang-python";
import { slugify } from "./lesson.ts";

export function tex(src: string, display: boolean): string {
  return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: "ignore", output: "htmlAndMathml" });
}

const escapeHtml = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

export function highlightPython(code: string): string {
  let html = "";
  highlightCode(code, pythonLanguage.parser.parse(code), classHighlighter,
    (text, classes) => { html += classes ? `<span class="${classes}">${escapeHtml(text)}</span>` : escapeHtml(text); },
    () => { html += "\n"; });
  return html;
}

const md = new Marked({
  gfm: true,
  extensions: [
    {
      name: "blockMath",
      level: "block",
      start: (src: string) => src.match(/^\$\$/m)?.index,
      tokenizer(src: string) {
        const m = /^\$\$([\s\S]+?)\$\$[ \t]*(?:\n|$)/.exec(src);
        if (m) return { type: "blockMath", raw: m[0], text: m[1].trim() };
      },
      renderer: (t: Tokens.Generic) => `<div class="math-display">${tex(t.text, true)}</div>\n`,
    },
    {
      name: "inlineMath",
      level: "inline",
      start: (src: string) => { const i = src.search(/(?<!\\)\$/); return i < 0 ? undefined : i; },
      tokenizer(src: string) {
        const m = /^\$((?:\\\$|[^$\n])+?)\$/.exec(src);
        if (m) return { type: "inlineMath", raw: m[0], text: m[1] };
      },
      renderer: (t: Tokens.Generic) => tex(t.text, false),
    },
  ],
  renderer: {
    heading({ tokens, depth, text }) {
      const inner = this.parser.parseInline(tokens);
      const id = slugify(text);
      return depth === 2
        ? `<h2 id="${id}">${inner}</h2>\n`
        : `<h${depth}>${inner}</h${depth}>\n`;
    },
    code({ text, lang }) {
      const body = lang === "python" || lang === "py" ? highlightPython(text) : escapeHtml(text);
      return `<pre class="code"><code>${body}</code></pre>\n`;
    },
    link({ href, title, tokens }) {
      const inner = this.parser.parseInline(tokens);
      const ext = /^https?:/.test(href);
      return `<a href="${href}"${title ? ` title="${escapeHtml(title)}"` : ""}${ext ? ' target="_blank" rel="noopener"' : ""}>${inner}</a>`;
    },
  },
});

export const renderMarkdown = (src: string): string => md.parse(src, { async: false }) as string;
export const renderInline = (src: string): string => md.parseInline(src, { async: false }) as string;
