import { Camera, CameraOff, Pause, Play, Power, ShieldCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { CameraInfo, CameraStatus, TrackedObject } from "../services/types";
import { Badge, Button, Card, Select } from "./ui";

function DetectionOverlay({ objects }: { objects: TrackedObject[] }) {
  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden="true">
      {objects.map((o) => {
        const [x1, y1, x2, y2] = o.box;
        const confirmed = o.status === "confirmed";
        const warn = confirmed && (o.approaching || (o.in_path && o.distance === "near"));
        const color = warn ? "var(--box-warn)" : confirmed ? "var(--box)" : "rgba(235,233,226,0.75)";
        return (
          <div
            key={o.track_id}
            className="absolute rounded-md transition-[left,top,width,height] duration-150 ease-linear"
            style={{
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              border: `${confirmed ? 2.5 : 1.5}px ${confirmed ? "solid" : "dashed"} ${color}`,
              boxShadow: confirmed ? "0 0 0 1px rgba(0,0,0,0.35)" : "none",
            }}
          >
            <span
              className={`absolute -top-px left-0 -translate-y-full whitespace-nowrap rounded-t-md px-2 py-0.5 text-[0.8rem] font-bold text-[#141816] ${confirmed ? "" : "opacity-80"}`}
              style={{ background: color }}
            >
              {confirmed ? o.spoken_label : `possible ${o.spoken_label}?`} · {Math.round(o.confidence * 100)}%
              <span className="ml-1.5 font-mono text-[0.7rem] opacity-70">#{o.track_id}</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function CameraPanel({
  camera,
  objects,
  frame,
  cameras,
  selectedIndex,
  onSelect,
  onStart,
  onPause,
  onResume,
  onStop,
  busy,
  showBoxes,
  onToggleBoxes,
}: {
  camera: CameraStatus | undefined;
  objects: TrackedObject[];
  frame: { width: number; height: number; inferenceMs: number; fps: number };
  cameras: CameraInfo[];
  selectedIndex: number;
  onSelect: (i: number) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  busy: boolean;
  showBoxes: boolean;
  onToggleBoxes: () => void;
}) {
  const state = camera?.state ?? "stopped";
  const active = state === "running" || state === "paused";
  const [streamKey, setStreamKey] = useState(0);

  // Re-open the MJPEG stream whenever the camera goes from off to running.
  const prevState = useRef(state);
  useEffect(() => {
    const was = prevState.current;
    if (state === "running" && was !== "running" && was !== "paused") setStreamKey((k) => k + 1);
    prevState.current = state;
  }, [state]);

  const ratio = camera?.resolution ? camera.resolution[0] / camera.resolution[1] : frame.width / frame.height || 4 / 3;
  const stateText: Record<string, string> = {
    stopped: "Camera off",
    starting: "Starting camera…",
    running: "Live",
    paused: "Paused",
    error: "Camera problem",
  };

  return (
    <Card
      title="Camera"
      icon={<Camera size={20} />}
      actions={
        <Badge tone={state === "running" ? "sage" : state === "error" ? "danger" : state === "paused" ? "warn" : "neutral"}>
          {state === "running" && <span className="live-dot inline-block h-2 w-2 rounded-full bg-sage" />}
          {stateText[state]}
        </Badge>
      }
    >
      <div
        className="relative w-full overflow-hidden rounded-xl border border-line bg-[#101312]"
        style={{ aspectRatio: String(ratio) }}
      >
        {active ? (
          <>
            <img
              key={streamKey}
              src={`/api/stream.mjpg?k=${streamKey}`}
              alt={(() => {
                const n = objects.filter((o) => o.status === "confirmed").length;
                return `Live camera preview. ${n} confirmed object${n === 1 ? "" : "s"}. The list of objects is in the Detected objects panel.`;
              })()}
              className="absolute inset-0 h-full w-full object-fill"
            />
            {showBoxes && <DetectionOverlay objects={objects} />}
            {state === "paused" && (
              <div className="absolute inset-0 grid place-items-center bg-black/45 text-lg font-bold text-white">
                Paused
              </div>
            )}
          </>
        ) : (
          <div className="absolute inset-0 grid place-items-center p-6 text-center text-[#c9cec8]">
            <div className="max-w-sm">
              {state === "error" ? <CameraOff className="mx-auto mb-3" size={36} aria-hidden /> : <Camera className="mx-auto mb-3 opacity-70" size={36} aria-hidden />}
              <p className="text-lg font-bold">{state === "error" ? "The camera couldn't start" : state === "starting" ? "Starting camera…" : "Camera is off"}</p>
              <p className="mt-1 text-sm opacity-80">
                {state === "error"
                  ? camera?.error
                  : "Turn the camera on to start detecting objects. Video never leaves this computer."}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-end gap-3">
        {!active ? (
          <Button variant="primary" size="lg" icon={<Power size={20} aria-hidden />} onClick={onStart} disabled={busy || state === "starting"} shortcut="C">
            {state === "starting" ? "Starting…" : "Start camera"}
          </Button>
        ) : (
          <>
            {state === "paused" ? (
              <Button variant="primary" size="lg" icon={<Play size={20} aria-hidden />} onClick={onResume} shortcut="P">
                Resume
              </Button>
            ) : (
              <Button size="lg" icon={<Pause size={20} aria-hidden />} onClick={onPause} shortcut="P">
                Pause
              </Button>
            )}
            <Button size="lg" icon={<Power size={20} aria-hidden />} onClick={onStop} shortcut="C">
              Stop camera
            </Button>
          </>
        )}
        {cameras.length > 1 && (
          <div className="min-w-56 flex-1">
            <Select
              label="Camera source"
              value={selectedIndex}
              options={cameras.map((c) => ({ value: c.index, label: c.name }))}
              onChange={onSelect}
            />
          </div>
        )}
        <label className="ml-auto inline-flex min-h-11 cursor-pointer items-center gap-2 text-sm text-muted">
          <input type="checkbox" checked={showBoxes} onChange={onToggleBoxes} className="h-5 w-5 accent-[var(--sage)]" />
          Show boxes
        </label>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-muted">
        <span className="inline-flex items-center gap-1.5">
          <ShieldCheck size={16} aria-hidden /> Processed locally · nothing is recorded or uploaded
        </span>
        {active && (
          <span className="tabular">
            Camera {camera?.fps ?? 0} fps · detection {frame.fps} fps · {frame.inferenceMs} ms
          </span>
        )}
      </div>
    </Card>
  );
}
