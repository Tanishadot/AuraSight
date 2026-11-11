import { Keyboard, Moon, Settings2, ShieldAlert, Sun, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ActivityEvent, CameraStatus } from "../services/types";
import { Button } from "./ui";

export function Header({
  camera,
  connected,
  theme,
  onToggleTheme,
  onShortcuts,
  onSettings,
}: {
  camera: CameraStatus | undefined;
  connected: boolean;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onShortcuts: () => void;
  onSettings: () => void;
}) {
  const live = camera?.state === "running";
  const paused = camera?.state === "paused";
  return (
    <header className="border-b border-line bg-surface/80 backdrop-blur">
      <div className="mx-auto flex max-w-[92rem] flex-wrap items-center gap-4 px-4 py-3 sm:px-8">
        <div className="flex items-center gap-3">
          <svg width="34" height="34" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="9" fill="var(--sage)" />
            <circle cx="16" cy="16" r="9.5" fill="none" stroke="var(--sage-contrast)" strokeWidth="1.6" opacity=".5" />
            <circle cx="16" cy="16" r="4.5" fill="var(--sage-contrast)" />
          </svg>
          <div>
            <h1 className="text-xl font-bold leading-none tracking-tight">AuraSight</h1>
            <p className="mt-1 text-xs text-muted">Private vision assistant · runs offline</p>
          </div>
        </div>

        <div
          className={`ml-2 inline-flex min-h-10 items-center gap-2 rounded-full border px-3.5 text-sm font-bold ${
            live ? "border-sage/40 bg-sage-soft text-sage-ink" : paused ? "border-warn/40 bg-warn-soft text-warn" : "border-line bg-surface-2 text-muted"
          }`}
          role="status"
        >
          <span className={`h-2.5 w-2.5 rounded-full ${live ? "live-dot bg-sage" : paused ? "bg-warn" : "bg-line-strong"}`} aria-hidden />
          {live ? "Camera on - processed on this device" : paused ? "Camera paused" : "Camera off"}
        </div>
        {!connected && (
          <div className="inline-flex min-h-10 items-center rounded-full border border-danger/40 bg-danger-soft px-3.5 text-sm font-bold text-danger" role="alert">
            Not connected to the local service
          </div>
        )}

        <nav className="ml-auto flex items-center gap-1.5" aria-label="App">
          <Button variant="quiet" size="icon" onClick={onToggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} aria-keyshortcuts="T">
            {theme === "dark" ? <Sun size={20} aria-hidden /> : <Moon size={20} aria-hidden />}
          </Button>
          <Button variant="quiet" size="icon" onClick={onShortcuts} aria-label="Keyboard shortcuts" aria-keyshortcuts="?">
            <Keyboard size={20} aria-hidden />
          </Button>
          <Button variant="secondary" onClick={onSettings} icon={<Settings2 size={18} aria-hidden />} aria-keyshortcuts=",">
            Settings
          </Button>
        </nav>
      </div>
    </header>
  );
}

const SAFETY_KEY = "aurasight-safety-ack";

export function SafetyNotice() {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(SAFETY_KEY) !== "1";
    } catch {
      return true;
    }
  });
  if (!open) return null;
  return (
    <div className="mx-auto mt-5 max-w-[92rem] px-4 sm:px-8">
      <div role="note" aria-label="Safety notice" className="flex items-start gap-3 rounded-2xl border border-warn/40 bg-warn-soft px-5 py-4 text-ink">
        <ShieldAlert className="mt-0.5 shrink-0 text-warn" size={22} aria-hidden />
        <div className="flex-1">
          <p className="font-bold">AuraSight is an aid, not a mobility aid.</p>
          <p className="mt-1 text-[0.95rem]">
            A webcam cannot reliably detect every obstacle, step, drop-off or hazard, and sees nothing outside its view.
            Keep using your white cane, guide dog or other mobility aid. Distances are rough estimates.
          </p>
        </div>
        <Button
          variant="quiet"
          size="icon"
          aria-label="Dismiss safety notice"
          onClick={() => {
            try {
              localStorage.setItem(SAFETY_KEY, "1");
            } catch {
              /* ignore */
            }
            setOpen(false);
          }}
        >
          <X size={18} aria-hidden />
        </Button>
      </div>
    </div>
  );
}

/** Screen-reader live regions: polite for routine events, assertive for warnings. */
export function LiveRegions({ latest, alert, enabled }: { latest: ActivityEvent | undefined; alert: ActivityEvent | null; enabled: boolean }) {
  const [polite, setPolite] = useState("");
  const [assertive, setAssertive] = useState("");
  const lastPolite = useRef<number>(0);
  const lastAlert = useRef<number>(0);

  useEffect(() => {
    if (!enabled || !latest || latest.id === lastPolite.current || latest.priority !== "info") return;
    if (latest.kind === "question") return; // the user just typed it
    lastPolite.current = latest.id;
    setPolite(latest.message);
  }, [latest, enabled]);

  useEffect(() => {
    if (!enabled || !alert || alert.id === lastAlert.current) return;
    lastAlert.current = alert.id;
    setAssertive(alert.message);
  }, [alert, enabled]);

  return (
    <>
      <div className="sr-only" aria-live="polite" aria-atomic="true">{polite}</div>
      <div className="sr-only" aria-live="assertive" aria-atomic="true">{assertive}</div>
    </>
  );
}
