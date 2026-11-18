// A CodeMirror 6 editor for Python with the site's own colours.
import { useEffect, useRef } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching, indentOnInput, syntaxHighlighting, HighlightStyle, indentUnit } from "@codemirror/language";
import { closeBrackets, closeBracketsKeymap } from "@codemirror/autocomplete";
import { python } from "@codemirror/lang-python";
import { tags } from "@lezer/highlight";

const style = HighlightStyle.define([
  { tag: [tags.keyword, tags.controlKeyword, tags.operatorKeyword, tags.definitionKeyword, tags.moduleKeyword], color: "var(--code-kw)" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--code-str)" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--code-num)" },
  { tag: [tags.comment, tags.lineComment], color: "var(--code-comment)", fontStyle: "italic" },
  { tag: [tags.function(tags.variableName), tags.function(tags.definition(tags.variableName))], color: "var(--code-fn)" },
  { tag: [tags.className, tags.definition(tags.className)], color: "var(--code-fn)" },
]);

export default function Editor({ value, onChange, onSubmit, label, minLines = 8 }: {
  value: string; onChange: (v: string) => void; onSubmit?: () => void; label: string; minLines?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const cb = useRef({ onChange, onSubmit });
  cb.current = { onChange, onSubmit };

  useEffect(() => {
    const v = new EditorView({
      parent: host.current!,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(), highlightActiveLineGutter(), history(), drawSelection(), indentOnInput(),
          bracketMatching(), closeBrackets(), highlightActiveLine(), python(), syntaxHighlighting(style),
          indentUnit.of("    "), EditorState.tabSize.of(4),
          keymap.of([
            { key: "Mod-Enter", run: () => { cb.current.onSubmit?.(); return true; } },
            ...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab,
          ]),
          EditorView.updateListener.of((u) => { if (u.docChanged) cb.current.onChange(u.state.doc.toString()); }),
          EditorView.contentAttributes.of({ "aria-label": label }),
          EditorView.theme({ ".cm-content, .cm-gutter": { minHeight: `${minLines * 1.5}em` } }),
        ],
      }),
    });
    view.current = v;
    return () => v.destroy();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // replace the text when the parent resets it (reset, show solution)
  useEffect(() => {
    const v = view.current;
    if (v && v.state.doc.toString() !== value) v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
  }, [value]);

  return <div className="editor" ref={host} />;
}
