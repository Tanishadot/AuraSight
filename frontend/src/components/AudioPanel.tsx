import { BellOff, BellRing, Square, Volume2, VolumeX } from "lucide-react";
import type { Settings, Voice } from "../services/types";
import type { SpeechState } from "../hooks/useAuraSocket";
import { Button, Card, Select, Slider, Toggle } from "./ui";

export function AudioPanel({
  settings,
  speech,
  voices,
  onChange,
  onStop,
  onTest,
}: {
  settings: Settings;
  speech: SpeechState;
  voices: Voice[];
  onChange: (patch: Partial<Settings>) => void;
  onStop: () => void;
  onTest: () => void;
}) {
  return (
    <Card
      title="Voice & announcements"
      icon={settings.speech_enabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
    >
      <div className="mb-3 flex min-h-[3.25rem] items-center gap-3 rounded-xl bg-surface-2 px-4 py-2.5" aria-hidden="true">
        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${speech.speaking ? "bg-sage live-dot" : "bg-line-strong"}`} />
        <p className="line-clamp-2 text-sm text-muted">
          {speech.speaking ? <>Speaking: <span className="text-ink">“{speech.text}”</span></> : "Quiet"}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <Button variant="danger" className="min-h-13 justify-center" onClick={onStop} shortcut="S" icon={<Square size={18} aria-hidden />}>
          Stop speaking
        </Button>
        <Button
          className="min-h-13 justify-center"
          aria-pressed={settings.announcements_muted}
          onClick={() => onChange({ announcements_muted: !settings.announcements_muted })}
          shortcut="M"
          icon={settings.announcements_muted ? <BellOff size={18} aria-hidden /> : <BellRing size={18} aria-hidden />}
        >
          {settings.announcements_muted ? "Unmute alerts" : "Mute alerts"}
        </Button>
      </div>

      <div className="mt-3 divide-y divide-line">
        <Toggle
          label="Speak out loud"
          description="Uses this computer's offline voice. Turn off if you prefer your screen reader."
          checked={settings.speech_enabled}
          onChange={(v) => onChange({ speech_enabled: v })}
        />
        <div className="py-2">
          <Slider label="Speech rate" value={settings.speech_rate} min={0.5} max={2} step={0.1} format={(v) => `${v.toFixed(1)}×`} onChange={(v) => onChange({ speech_rate: Math.round(v * 10) / 10 })} />
          <Slider label="Volume" value={settings.speech_volume} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => onChange({ speech_volume: Math.round(v * 100) / 100 })} />
          {voices.length > 1 && (
            <Select
              label="Voice"
              value={settings.voice_id ?? voices[0].id}
              options={voices.map((v) => ({ value: v.id, label: v.name }))}
              onChange={(v) => onChange({ voice_id: v })}
            />
          )}
          <Button className="mt-2" onClick={onTest}>Test voice</Button>
        </div>
      </div>
    </Card>
  );
}
