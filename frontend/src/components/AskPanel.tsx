import { CornerDownLeft, MessagesSquare } from "lucide-react";
import { forwardRef, useState } from "react";
import type { AnswerResult } from "../services/types";
import { Badge, Button, Card } from "./ui";

const SUGGESTIONS = [
  "What is in front of me?",
  "Where is my water bottle?",
  "Is there a chair nearby?",
  "Describe this room.",
  "What objects are on the table?",
];

export interface Turn {
  id: number;
  question: string;
  answer?: AnswerResult;
  error?: string;
}

export const AskPanel = forwardRef<HTMLInputElement, {
  turns: Turn[];
  pending: boolean;
  cameraActive: boolean;
  onAsk: (q: string) => void;
}>(function AskPanel({ turns, pending, cameraActive, onAsk }, inputRef) {
  const [q, setQ] = useState("");
  const submit = (text: string) => {
    const t = text.trim();
    if (!t || pending) return;
    onAsk(t);
    setQ("");
  };
  return (
    <Card title="Ask about your surroundings" icon={<MessagesSquare size={20} />}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(q);
        }}
        className="flex gap-2"
      >
        <label htmlFor="ask-input" className="sr-only">
          Ask a question about what the camera sees
        </label>
        <input
          id="ask-input"
          ref={inputRef}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={cameraActive ? "e.g. Where is my water bottle?" : "Start the camera to ask questions"}
          autoComplete="off"
          aria-keyshortcuts="A /"
          maxLength={500}
          className="min-h-14 flex-1 rounded-xl border border-line-strong bg-surface px-4 text-lg text-ink placeholder:text-muted"
        />
        <Button type="submit" variant="primary" size="lg" disabled={pending || !q.trim()} icon={<CornerDownLeft size={20} aria-hidden />}>
          {pending ? "Thinking…" : "Ask"}
        </Button>
      </form>

      <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Example questions">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={pending || !cameraActive}
            onClick={() => submit(s)}
            className="min-h-10 rounded-full border border-line bg-surface-2 px-3.5 text-sm text-ink hover:border-sage disabled:opacity-45"
          >
            {s}
          </button>
        ))}
      </div>

      {turns.length > 0 && (
        <ol className="mt-5 space-y-4" aria-label="Conversation, newest first">
          {turns.map((t) => (
            <li key={t.id} className="fade-up">
              <p className="text-sm font-bold text-muted">You asked: <span className="font-normal text-ink">{t.question}</span></p>
              {t.answer ? (
                <div className="mt-1.5 rounded-xl border border-line bg-surface-2 px-4 py-3">
                  <p className="text-lg leading-relaxed">{t.answer.answer}</p>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted">
                    {t.answer.grounded && <Badge tone="sage">Seen by detector</Badge>}
                    {t.answer.vlm_used && <Badge tone={t.answer.grounded ? "neutral" : "warn"}>Scene model{t.answer.grounded ? "" : " · may be less reliable"}</Badge>}
                    <span className="tabular">{(t.answer.latency_ms / 1000).toFixed(1)} s</span>
                  </div>
                </div>
              ) : t.error ? (
                <p className="mt-1.5 text-danger" role="alert">{t.error}</p>
              ) : (
                <p className="mt-1.5 breathe text-muted" role="status">Thinking…</p>
              )}
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
});
