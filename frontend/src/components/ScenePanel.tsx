import { Eye, Sparkles } from "lucide-react";
import type { SceneResult, VlmStatus } from "../services/types";
import { Badge, Button, Card } from "./ui";

export function ScenePanel({
  scene,
  progress,
  vlm,
  cameraActive,
  pending,
  error,
  onDescribe,
}: {
  scene: (SceneResult & { ts: number }) | null;
  progress: { stage: "idle" | "analysing" | "streaming"; text: string };
  vlm: VlmStatus | undefined;
  cameraActive: boolean;
  pending: boolean;
  error: string | null;
  onDescribe: () => void;
}) {
  const vlmUsable = vlm && (vlm.state === "ready" || vlm.state === "busy");
  return (
    <Card
      title="Scene"
      icon={<Eye size={20} />}
      actions={
        <Button
          variant="primary"
          size="lg"
          onClick={onDescribe}
          disabled={!cameraActive || pending}
          shortcut="D"
          icon={<Sparkles size={20} aria-hidden />}
        >
          {pending ? "Looking…" : "Describe scene"}
        </Button>
      }
    >
      <div aria-busy={pending} className="min-h-[6.5rem]">
        {(pending || progress.stage !== "idle") && progress.text ? (
          <div role="status">
            <p className="text-[1.3rem] leading-relaxed text-ink">
              {progress.text}
              <span className="breathe ml-1 inline-block h-5 w-2 translate-y-0.5 rounded-sm bg-sage" aria-hidden />
            </p>
          </div>
        ) : pending || progress.stage === "analysing" ? (
          <div className="space-y-3" role="status">
            <p className="breathe text-lg text-muted">
              Looking at the scene… the first sentence usually takes about 10 seconds on this computer.
            </p>
            <div className="h-3 w-4/5 rounded-full bg-surface-2 breathe" />
            <div className="h-3 w-3/5 rounded-full bg-surface-2 breathe" />
          </div>
        ) : error ? (
          <p className="text-lg text-danger" role="alert">{error}</p>
        ) : scene ? (
          <figure className="fade-up">
            <blockquote className="text-[1.3rem] leading-relaxed text-ink">{scene.description}</blockquote>
            <figcaption className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted">
              {scene.source === "vlm" ? (
                <Badge tone="sage">Scene model + detector</Badge>
              ) : (
                <Badge tone="warn">Object detector only</Badge>
              )}
              {scene.auto && <Badge>Automatic</Badge>}
              {scene.latency_ms > 0 && (
                <span className="tabular">
                  {scene.first_sentence_ms ? `first words ${(scene.first_sentence_ms / 1000).toFixed(1)} s · ` : ""}
                  complete {(scene.latency_ms / 1000).toFixed(1)} s
                </span>
              )}
              <span>·</span>
              <time>{new Date(scene.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
            </figcaption>
          </figure>
        ) : (
          <p className="text-lg text-muted">
            {cameraActive
              ? "Press Describe scene (or the D key) for a spoken description of what the camera sees."
              : "Turn the camera on, then ask for a description of your surroundings."}
          </p>
        )}
      </div>
      {!vlmUsable && vlm && (
        <p className="mt-4 rounded-xl bg-surface-2 px-3 py-2 text-sm text-muted">
          The scene model is {vlm.state === "starting" ? "still loading" : vlm.state === "disabled" ? "turned off in settings" : "not available"}
          {vlm.detail ? ` (${vlm.detail})` : ""}. Descriptions will fall back to the object detector.
        </p>
      )}
    </Card>
  );
}
