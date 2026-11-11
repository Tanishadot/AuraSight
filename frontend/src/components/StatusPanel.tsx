import { Activity, AlertTriangle, CheckCircle2, CircleDashed, Loader2, MinusCircle } from "lucide-react";
import type { SystemStatus } from "../services/types";
import { Button, Card } from "./ui";

type Tone = "ok" | "busy" | "off" | "bad";

function Row({ name, tone, text, detail }: { name: string; tone: Tone; text: string; detail?: string }) {
  const icon = {
    ok: <CheckCircle2 size={18} className="text-sage" aria-hidden />,
    busy: <Loader2 size={18} className="animate-spin text-warn motion-reduce:animate-none" aria-hidden />,
    off: <MinusCircle size={18} className="text-muted" aria-hidden />,
    bad: <AlertTriangle size={18} className="text-danger" aria-hidden />,
  }[tone];
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className="mt-0.5">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-3">
          <span className="font-bold">{name}</span>
          <span className={`text-sm font-bold ${tone === "bad" ? "text-danger" : tone === "ok" ? "text-sage-ink" : "text-muted"}`}>
            {text}
          </span>
        </div>
        {detail && <p className="mt-0.5 break-words text-sm text-muted">{detail}</p>}
      </div>
    </li>
  );
}

export function StatusPanel({
  status,
  connected,
  onRetryVlm,
}: {
  status: SystemStatus | null;
  connected: boolean;
  onRetryVlm: () => void;
}) {
  if (!connected || !status) {
    return (
      <Card title="System" icon={<Activity size={20} />}>
        <ul>
          <Row
            name="AuraSight service"
            tone="bad"
            text="Not connected"
            detail="Waiting for the local backend on this computer. Start it with scripts/dev.ps1."
          />
        </ul>
      </Card>
    );
  }
  const { camera, detector, vlm, speech } = status;
  const cam: [Tone, string] =
    camera.state === "running" ? ["ok", "Live"] :
    camera.state === "paused" ? ["off", "Paused"] :
    camera.state === "starting" ? ["busy", "Starting"] :
    camera.state === "error" ? ["bad", "Problem"] : ["off", "Off"];
  const det: [Tone, string] =
    detector.state === "ready" ? ["ok", "Ready"] :
    detector.state === "loading" ? ["busy", "Loading"] :
    detector.state === "error" ? ["bad", "Unavailable"] : ["off", "Loads with camera"];
  const v: [Tone, string] =
    vlm.state === "ready" ? ["ok", "Ready"] :
    vlm.state === "busy" ? ["busy", "Thinking"] :
    vlm.state === "starting" ? ["busy", "Loading"] :
    vlm.state === "disabled" ? ["off", "Disabled"] :
    vlm.state === "not_started" ? ["off", "Not started"] : ["bad", "Unavailable"];
  const sp: [Tone, string] =
    speech.state === "speaking" ? ["ok", "Speaking"] :
    speech.state === "ready" ? ["ok", "Ready"] :
    speech.state === "error" ? ["bad", "Unavailable"] : ["busy", "Starting"];

  const vlmDetail = [vlm.model, vlm.last_latency_ms ? `last answer ${(vlm.last_latency_ms / 1000).toFixed(1)} s` : "", vlm.detail]
    .filter(Boolean)
    .join(" · ");

  return (
    <Card
      title="System"
      icon={<Activity size={20} />}
      actions={<span className="text-sm text-muted inline-flex items-center gap-1.5"><CircleDashed size={14} aria-hidden /> Offline · on-device</span>}
    >
      <ul className="divide-y divide-line">
        <Row name="Camera" tone={cam[0]} text={cam[1]} detail={camera.error ?? undefined} />
        <Row
          name="Object detector"
          tone={det[0]}
          text={det[1]}
          detail={detector.error ?? `${detector.model}${detector.inference_ms ? ` · ${detector.inference_ms} ms per frame` : ""}`}
        />
        <Row name="Scene model" tone={v[0]} text={v[1]} detail={vlmDetail || undefined} />
        <Row name="Voice" tone={sp[0]} text={sp[1]} detail={speech.error ?? (speech.engine ? `Offline engine: ${speech.engine}` : undefined)} />
      </ul>
      {(vlm.state === "unavailable" || vlm.state === "error" || vlm.state === "not_started") && (
        <Button className="mt-3 w-full justify-center" onClick={onRetryVlm}>
          Try starting the scene model
        </Button>
      )}
    </Card>
  );
}
