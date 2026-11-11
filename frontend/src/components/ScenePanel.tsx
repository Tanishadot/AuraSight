import { Eye, RotateCcw, Sparkles, Square, VolumeX } from "lucide-react";
import type { SpeechState } from "../hooks/useAuraSocket";
import type { SceneResult, VlmStatus } from "../services/types";
import { Badge, Button, Card } from "./ui";

/** Three animated sage bars; static under prefers-reduced-motion. */
function SpeakingBars() {
  return (
    <span className="inline-flex h-4 items-end gap-[3px]" aria-hidden>
      {[0, 1, 2].map((i) => (
        <span key={i} className="eq-bar h-full w-[3px] rounded-full bg-sage" />
      ))}
    </span>
  );
}

export function ScenePanel({
  scene,
  progress,
  vlm,
  cameraActive,
  pending,
  error,
  speech,
  speechEnabled,
  onDescribe,
  onReplay,
  onStopSpeaking,
}: {
  scene: (SceneResult & { ts: number }) | null;
  progress: { stage: "idle" | "analysing" | "streaming"; text: string };
  vlm: VlmStatus | undefined;
  cameraActive: boolean;
  pending: boolean;
  error: string | null;
  speech: SpeechState;
  speechEnabled: boolean;
  onDescribe: () => void;
  onReplay: () => void;
  onStopSpeaking: () => void;
}) {
  const vlmUsable = vlm && (vlm.state === "ready" || vlm.state === "busy");
  const reading = speech.speaking && speech.kind === "description";
  const ttsError = speech.error && speech.error.kind === "description" ? speech.error.message : null;
  const generating = pending || progress.stage !== "idle";
  const showControls = !!scene || generating;

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
        {generating && progress.text ? (
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

      {showControls && (
        <div
          role="group"
          aria-label="Description audio"
          className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-4"
        >
          <Button
            variant="secondary"
            onClick={onReplay}
            disabled={!scene || generating || !speechEnabled}
            shortcut="R"
            icon={<RotateCcw size={18} aria-hidden />}
            aria-label="Read the description aloud again"
            title={speechEnabled ? "Read the description aloud again (R)" : "Speak out loud is off"}
          >
            Replay
          </Button>
          <Button
            variant={reading ? "danger" : "secondary"}
            onClick={onStopSpeaking}
            disabled={!reading && !generating}
            shortcut="S"
            icon={<Square size={16} aria-hidden />}
            aria-label="Stop speaking"
            title="Stop speaking (S or Esc)"
          >
            Stop
          </Button>
          <p className="flex min-h-6 items-center gap-2 text-sm" role="status" aria-live="polite">
            {reading ? (
              <>
                <SpeakingBars />
                <span className="font-bold text-sage-ink">Reading the description aloud…</span>
              </>
            ) : !speechEnabled ? (
              <>
                <VolumeX size={16} className="text-muted" aria-hidden />
                <span className="text-muted">Speak out loud is off. The description is shown, not spoken.</span>
              </>
            ) : null}
          </p>
        </div>
      )}
      {ttsError && (
        <p className="mt-3 rounded-xl border border-danger/40 bg-danger-soft px-3 py-2 text-sm text-danger" role="alert">
          The description couldn't be read aloud ({ttsError}). The text above is complete; try Replay.
        </p>
      )}

      {!vlmUsable && vlm && (
        <p className="mt-4 rounded-xl bg-surface-2 px-3 py-2 text-sm text-muted">
          The scene model is {vlm.state === "starting" ? "still loading" : vlm.state === "disabled" ? "turned off in settings" : "not available"}
          {vlm.detail ? ` (${vlm.detail})` : ""}. Descriptions will fall back to the object detector.
        </p>
      )}
    </Card>
  );
}
