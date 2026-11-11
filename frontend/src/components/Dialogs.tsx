import { Keyboard, Settings2, X } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef } from "react";
import { SHORTCUT_HELP } from "../hooks/useShortcuts";
import type { Settings } from "../services/types";
import { Button, Select, Slider, Toggle } from "./ui";

/** Native <dialog>: focus trapping, Esc to close and inert background for free. */
function Modal({ open, onClose, title, icon, children }: { open: boolean; onClose: () => void; title: string; icon: ReactNode; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-labelledby="dlg-title"
      className="m-auto max-h-[88vh] w-[min(42rem,94vw)] overflow-y-auto rounded-2xl border border-line bg-surface p-0 text-ink shadow-2xl backdrop:bg-[var(--overlay)]"
    >
      <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-surface px-6 py-4">
        <span className="text-sage-ink" aria-hidden>{icon}</span>
        <h2 id="dlg-title" className="text-xl font-bold">{title}</h2>
        <Button variant="quiet" size="icon" className="ml-auto" onClick={onClose} aria-label="Close">
          <X size={20} aria-hidden />
        </Button>
      </div>
      <div className="px-6 py-5">{children}</div>
    </dialog>
  );
}

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Modal open={open} onClose={onClose} title="Keyboard shortcuts" icon={<Keyboard size={22} />}>
      <p className="mb-4 text-muted">Single keys work anywhere except while typing in a text box. Esc always stops speech.</p>
      <table className="w-full">
        <thead className="sr-only">
          <tr><th>Key</th><th>Action</th></tr>
        </thead>
        <tbody className="divide-y divide-line">
          {SHORTCUT_HELP.map((s) => (
            <tr key={s.keys}>
              <td className="w-28 py-2.5 pr-4 align-top">
                <kbd className="rounded-lg border border-line-strong bg-surface-2 px-2 py-1 font-mono text-sm">{s.keys}</kbd>
              </td>
              <td className="py-2.5">{s.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Modal>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="mb-6">
      <legend className="mb-1 text-sm font-bold uppercase tracking-wider text-muted">{title}</legend>
      <div className="divide-y divide-line rounded-xl border border-line px-4">{children}</div>
    </fieldset>
  );
}

export function SettingsDialog({
  open,
  onClose,
  settings,
  onChange,
}: {
  open: boolean;
  onClose: () => void;
  settings: Settings;
  onChange: (p: Partial<Settings>) => void;
}) {
  return (
    <Modal open={open} onClose={onClose} title="Settings" icon={<Settings2 size={22} />}>
      <Section title="Announcements">
        <Toggle label="Announce new objects" checked={settings.announce_new_objects} onChange={(v) => onChange({ announce_new_objects: v })} />
        <Toggle
          label="Obstacle and approach warnings"
          description="Warns when something is close in the centre of the view or getting closer quickly."
          checked={settings.obstacle_alerts}
          onChange={(v) => onChange({ obstacle_alerts: v })}
        />
        <Toggle label="Announce when objects leave" checked={settings.announce_departures} onChange={(v) => onChange({ announce_departures: v })} />
        <Toggle label="Announce big scene changes" checked={settings.announce_scene_changes} onChange={(v) => onChange({ announce_scene_changes: v })} />
        <div className="py-2">
          <Select
            label="Detail level"
            value={settings.verbosity}
            options={[
              { value: "brief", label: "Brief - object and direction" },
              { value: "detailed", label: "Detailed - also approximate distance" },
            ]}
            onChange={(v) => onChange({ verbosity: v })}
          />
          <Slider
            label="Don't repeat the same kind of object for"
            value={settings.announce_cooldown_s}
            min={0}
            max={120}
            step={5}
            format={(v) => `${v} seconds`}
            onChange={(v) => onChange({ announce_cooldown_s: v })}
          />
        </div>
        <Toggle
          label="Screen reader announcements"
          description="Also post events to your screen reader. Turn off if you hear everything twice."
          checked={settings.screen_reader_announcements}
          onChange={(v) => onChange({ screen_reader_announcements: v })}
        />
      </Section>

      <Section title="Scene model">
        <Toggle
          label="Describe automatically"
          description="Periodically describes the scene and after big changes. Uses more CPU."
          checked={settings.auto_describe}
          onChange={(v) => onChange({ auto_describe: v })}
        />
        <div className="py-2">
          <Slider
            label="Automatic description interval"
            value={settings.auto_describe_interval_s}
            min={15}
            max={300}
            step={15}
            format={(v) => `every ${v} s`}
            onChange={(v) => onChange({ auto_describe_interval_s: v })}
          />
          <Select
            label="Runtime"
            hint="llama.cpp with SmolVLM is bundled. Ollama must be installed separately."
            value={settings.vlm_backend}
            options={[
              { value: "llamacpp", label: "llama.cpp - SmolVLM 500M (recommended)" },
              { value: "ollama", label: "Ollama - e.g. moondream" },
              { value: "none", label: "Off - object detector only" },
            ]}
            onChange={(v) => onChange({ vlm_backend: v })}
          />
        </div>
      </Section>

      <Section title="Performance">
        <div className="py-2">
          <Select
            label="Detection resolution"
            hint="Lower is faster; higher finds smaller and farther objects."
            value={settings.inference_size}
            options={[256, 320, 416, 480, 512, 640].map((v) => ({ value: v as Settings["inference_size"], label: `${v} px${v === 416 ? " (default)" : ""}` }))}
            onChange={(v) => onChange({ inference_size: v })}
          />
          <Slider label="Detections per second" value={settings.detection_fps} min={1} max={15} step={1} format={(v) => `${v} per second`} onChange={(v) => onChange({ detection_fps: v })} />
          <Slider
            label="Detection confidence"
            value={settings.confidence_threshold}
            min={0.2}
            max={0.8}
            step={0.05}
            format={(v) => `${Math.round(v * 100)}%`}
            onChange={(v) => onChange({ confidence_threshold: Math.round(v * 100) / 100 })}
          />
        </div>
      </Section>

      <Section title="Appearance">
        <div className="py-2">
          <Select
            label="Theme"
            value={settings.theme}
            options={[
              { value: "system", label: "Match system" },
              { value: "light", label: "Light" },
              { value: "dark", label: "Dark" },
            ]}
            onChange={(v) => onChange({ theme: v })}
          />
        </div>
      </Section>
    </Modal>
  );
}
