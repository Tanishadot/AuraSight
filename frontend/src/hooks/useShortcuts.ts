import { useEffect, useRef } from "react";

export interface Shortcut {
  keys: string[]; // KeyboardEvent.key values (case-insensitive for letters)
  label: string;
  action: () => void;
  /** also fire while typing in a text field */
  global?: boolean;
}

export const SHORTCUT_HELP: { keys: string; label: string }[] = [
  { keys: "D", label: "Describe the scene (scene model)" },
  { keys: "R", label: "Read the last scene description again" },
  { keys: "W", label: "What's around me? (instant, from the object detector)" },
  { keys: "A or /", label: "Ask a question" },
  { keys: "S or Esc", label: "Stop speaking" },
  { keys: "C", label: "Turn camera on or off" },
  { keys: "P", label: "Pause or resume the camera" },
  { keys: "M", label: "Mute or unmute automatic announcements" },
  { keys: "T", label: "Switch light / dark theme" },
  { keys: ",", label: "Open settings" },
  { keys: "?", label: "Show keyboard shortcuts" },
];

function isTyping(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

/** Single-key shortcuts that never fire while the user is typing or holding modifiers. */
export function useShortcuts(shortcuts: Shortcut[], enabled = true) {
  const ref = useRef(shortcuts);
  ref.current = shortcuts;

  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      const typing = isTyping(e.target);
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      for (const s of ref.current) {
        if (!s.keys.some((k) => (k.length === 1 ? k.toLowerCase() : k) === key)) continue;
        if (typing && !s.global) continue;
        e.preventDefault();
        s.action();
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
