import { AlertTriangle, Bell, Camera, CircleHelp, Eye, LogIn, LogOut, MessageSquare, Shuffle, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import type { ActivityEvent } from "../services/types";
import { Card } from "./ui";

const kindMeta: Record<string, { icon: ReactNode; label: string }> = {
  entered: { icon: <LogIn size={16} />, label: "New object" },
  left: { icon: <LogOut size={16} />, label: "Left view" },
  obstacle: { icon: <TriangleAlert size={16} />, label: "Obstacle" },
  approaching: { icon: <AlertTriangle size={16} />, label: "Approaching" },
  scene_change: { icon: <Shuffle size={16} />, label: "Scene changed" },
  description: { icon: <Eye size={16} />, label: "Scene" },
  question: { icon: <CircleHelp size={16} />, label: "You asked" },
  answer: { icon: <MessageSquare size={16} />, label: "Answer" },
  summary: { icon: <Eye size={16} />, label: "Around you" },
  camera: { icon: <Camera size={16} />, label: "Camera" },
};

const filters = [
  { id: "all", label: "All" },
  { id: "alerts", label: "Alerts" },
  { id: "objects", label: "Objects" },
  { id: "conversation", label: "Conversation" },
] as const;

function matches(f: (typeof filters)[number]["id"], e: ActivityEvent) {
  if (f === "all") return true;
  if (f === "alerts") return e.priority !== "info";
  if (f === "objects") return ["entered", "left", "obstacle", "approaching", "scene_change"].includes(e.kind);
  return ["question", "answer", "description", "summary"].includes(e.kind);
}

export function ActivityFeed({ events }: { events: ActivityEvent[] }) {
  const [filter, setFilter] = useState<(typeof filters)[number]["id"]>("all");
  const shown = useMemo(() => events.filter((e) => matches(filter, e)).slice(0, 60), [events, filter]);
  return (
    <Card title="Activity" icon={<Bell size={20} />}>
      <div role="group" aria-label="Filter activity" className="mb-3 flex flex-wrap gap-1.5">
        {filters.map((f) => (
          <button
            key={f.id}
            type="button"
            aria-pressed={filter === f.id}
            onClick={() => setFilter(f.id)}
            className={`min-h-9 rounded-full border px-3 text-sm font-bold ${
              filter === f.id ? "border-sage bg-sage-soft text-sage-ink" : "border-line bg-transparent text-muted hover:bg-surface-2"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>
      {shown.length === 0 ? (
        <p className="text-muted">Nothing yet. Events such as new objects and alerts will be listed here.</p>
      ) : (
        <ol className="max-h-[22rem] space-y-1 overflow-y-auto pr-1" aria-label="Recent activity, newest first" tabIndex={0}>
          {shown.map((e) => {
            const meta = kindMeta[e.kind] ?? { icon: <Bell size={16} />, label: e.kind };
            const tone =
              e.priority === "error" ? "text-danger bg-danger-soft" : e.priority === "warning" ? "text-warn bg-warn-soft" : "text-sage-ink bg-sage-soft";
            return (
              <li key={e.id} className="fade-up flex gap-3 rounded-xl px-2 py-2 hover:bg-surface-2">
                <span className={`mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg ${tone}`} aria-hidden>
                  {meta.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-[0.95rem] leading-snug">
                    <span className="sr-only">{meta.label}: </span>
                    {e.message}
                  </p>
                  <p className="text-xs text-muted">
                    <span aria-hidden>{meta.label} · </span>
                    <time dateTime={new Date(e.ts * 1000).toISOString()}>
                      {new Date(e.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </time>
                  </p>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}
