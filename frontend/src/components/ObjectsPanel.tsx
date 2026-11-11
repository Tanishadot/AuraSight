import { ArrowDownToLine, Boxes, Navigation } from "lucide-react";
import type { TrackedObject } from "../services/types";
import { Badge, Button, Card } from "./ui";

const order = { near: 0, middle: 1, far: 2 } as const;

const byProximity = (a: TrackedObject, b: TrackedObject) =>
  order[a.distance] - order[b.distance] ||
  Math.abs(0.5 - (a.box[0] + a.box[2]) / 2) - Math.abs(0.5 - (b.box[0] + b.box[2]) / 2);

export function ObjectsPanel({
  objects,
  active,
  onSummary,
}: {
  objects: TrackedObject[];
  active: boolean;
  onSummary: () => void;
}) {
  const confirmed = objects.filter((o) => o.status === "confirmed").sort(byProximity);
  const tentative = objects.filter((o) => o.status === "tentative").sort(byProximity);
  return (
    <Card
      title="Detected objects"
      icon={<Boxes size={20} />}
      actions={
        <Button size="md" variant="secondary" onClick={onSummary} disabled={!active} shortcut="W" icon={<Navigation size={16} aria-hidden />}>
          Read out
        </Button>
      }
    >
      {!active ? (
        <p className="text-muted">Objects will appear here once the camera is on.</p>
      ) : confirmed.length === 0 ? (
        <p className="text-muted">No confirmed objects in view right now.</p>
      ) : (
        <ul className="divide-y divide-line" aria-label={`${confirmed.length} confirmed objects, nearest first`}>
          {confirmed.map((o) => {
            const warn = o.approaching || (o.in_path && o.distance === "near");
            return (
              <li key={o.track_id} className="flex items-center gap-3 py-2.5">
                <span
                  aria-hidden
                  className={`h-9 w-1.5 shrink-0 rounded-full ${warn ? "bg-warn" : o.distance === "near" ? "bg-sage" : o.distance === "middle" ? "bg-sage/60" : "bg-line-strong"}`}
                />
                <div className="min-w-0 flex-1">
                  <p className="font-bold capitalize">
                    {o.spoken_label}
                    <span className="sr-only">, </span>
                    <span className="ml-2 font-normal normal-case text-muted">{o.phrase}</span>
                  </p>
                  <p className="text-sm text-muted tabular">
                    {Math.round(o.confidence * 100)}% confident · ID {o.track_id} · in view {Math.round(o.age_s)} s
                  </p>
                </div>
                {o.approaching ? (
                  <Badge tone="warn"><ArrowDownToLine size={12} aria-hidden /> Approaching</Badge>
                ) : warn ? (
                  <Badge tone="warn">In your path</Badge>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}

      {active && tentative.length > 0 && (
        <details className="mt-3 rounded-xl border border-dashed border-line-strong px-3 py-2">
          <summary className="cursor-pointer text-sm font-bold text-muted">
            Not yet verified ({tentative.length}) - not announced
          </summary>
          <ul className="mt-2 space-y-1.5" aria-label="Unverified detections">
            {tentative.map((o) => (
              <li key={o.track_id} className="text-sm text-muted">
                <span className="font-bold">Possible {o.spoken_label}</span>, {o.phrase} · {Math.round(o.confidence * 100)}%
                <span className="block text-xs">{o.status_reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="mt-3 text-xs text-muted">
        Objects are confirmed only after they stay stable for about a second; easily-confused ones (TVs, fridges) are also
        re-checked on a zoomed view. Positions are approximate, not measured distances.
      </p>
    </Card>
  );
}
