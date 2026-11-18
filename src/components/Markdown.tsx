import { useMemo } from "react";
import { renderInline, renderMarkdown } from "../lib/markdown.ts";

export default function Markdown({ text, inline = false, className }: { text: string; inline?: boolean; className?: string }) {
  const html = useMemo(() => (inline ? renderInline(text) : renderMarkdown(text)), [text, inline]);
  return inline
    ? <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
    : <div className={className ?? "prose"} dangerouslySetInnerHTML={{ __html: html }} />;
}
