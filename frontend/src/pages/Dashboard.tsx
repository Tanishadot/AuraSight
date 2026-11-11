import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ActivityFeed } from "../components/ActivityFeed";
import { AskPanel, type Turn } from "../components/AskPanel";
import { AudioPanel } from "../components/AudioPanel";
import { CameraPanel } from "../components/CameraPanel";
import { Header, LiveRegions, SafetyNotice } from "../components/Chrome";
import { SettingsDialog, ShortcutsDialog } from "../components/Dialogs";
import { ObjectsPanel } from "../components/ObjectsPanel";
import { ScenePanel } from "../components/ScenePanel";
import { StatusPanel } from "../components/StatusPanel";
import { useAuraSocket } from "../hooks/useAuraSocket";
import { useShortcuts } from "../hooks/useShortcuts";
import { useTheme } from "../hooks/useTheme";
import { api, ApiError } from "../services/api";
import type { CameraInfo, Settings, Voice } from "../services/types";

export function Dashboard() {
  const live = useAuraSocket();
  const [local, setLocal] = useState<Partial<Settings>>({});
  const settings = useMemo(() => (live.settings ? { ...live.settings, ...local } : null), [live.settings, local]);
  const theme = useTheme(settings?.theme);

  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [camBusy, setCamBusy] = useState(false);
  const [describing, setDescribing] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [asking, setAsking] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const askRef = useRef<HTMLInputElement>(null);
  const turnId = useRef(0);
  const pending = useRef<Partial<Settings>>({});
  const flushTimer = useRef<number>();

  const camera = live.status?.camera;
  const camState = camera?.state ?? "stopped";
  const cameraActive = camState === "running" || camState === "paused";

  // -- data that isn't pushed over the socket ---------------------------------
  useEffect(() => {
    if (!live.connected) return;
    api.cameras().then(setCameras).catch(() => {});
  }, [live.connected]);
  useEffect(() => {
    if (live.status?.speech.state === "ready" && voices.length === 0) api.voices().then(setVoices).catch(() => {});
  }, [live.status?.speech.state, voices.length]);

  // Server echo wins once it arrives: drop local overrides it confirms.
  useEffect(() => {
    if (!live.settings) return;
    setLocal((cur) => {
      const next: Partial<Settings> = {};
      for (const [k, v] of Object.entries(cur)) {
        if ((live.settings as any)[k] !== v) (next as any)[k] = v;
      }
      return next;
    });
  }, [live.settings]);

  /** Optimistic settings update; network writes are debounced (sliders). */
  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setLocal((cur) => ({ ...cur, ...patch }));
    pending.current = { ...pending.current, ...patch };
    window.clearTimeout(flushTimer.current);
    flushTimer.current = window.setTimeout(() => {
      const body = pending.current;
      pending.current = {};
      api.updateSettings(body).catch((e: ApiError) => {
        setNotice(`Couldn't save settings: ${e.message}`);
        setLocal({});
      });
    }, 250);
  }, []);

  const flash = (msg: string) => {
    setNotice(msg);
    window.setTimeout(() => setNotice((n) => (n === msg ? null : n)), 6000);
  };

  // -- actions -------------------------------------------------------------
  const startCamera = useCallback(async (index?: number) => {
    setCamBusy(true);
    try {
      await api.startCamera(index);
    } catch (e) {
      flash((e as ApiError).message);
    } finally {
      setCamBusy(false);
    }
  }, []);
  const stopCamera = useCallback(() => api.stopCamera().catch((e: ApiError) => flash(e.message)), []);
  const togglePause = useCallback(() => {
    if (camState === "running") api.pauseCamera();
    else if (camState === "paused") api.resumeCamera();
  }, [camState]);
  const toggleCamera = useCallback(() => {
    if (cameraActive) stopCamera();
    else startCamera();
  }, [cameraActive, startCamera, stopCamera]);

  const describe = useCallback(async () => {
    if (!cameraActive) {
      flash("Turn the camera on first.");
      return;
    }
    if (describing) return;
    setDescribing(true);
    setSceneError(null);
    try {
      await api.describe();
    } catch (e) {
      const err = e as ApiError;
      setSceneError(err.status === 429 ? "Still working on the previous request - try again in a moment." : err.message);
    } finally {
      setDescribing(false);
    }
  }, [cameraActive, describing]);

  const ask = useCallback(async (question: string) => {
    const id = ++turnId.current;
    setTurns((t) => [{ id, question }, ...t].slice(0, 8));
    setAsking(true);
    try {
      const answer = await api.ask(question);
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, answer } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: (e as ApiError).message } : x)));
    } finally {
      setAsking(false);
    }
  }, []);

  const stopSpeech = useCallback(() => api.stopSpeech().catch(() => {}), []);

  useShortcuts(
    [
      { keys: ["Escape"], label: "stop", action: stopSpeech, global: true },
      { keys: ["s"], label: "stop", action: stopSpeech },
      { keys: ["d"], label: "describe", action: describe },
      { keys: ["w"], label: "summary", action: () => api.summary().catch(() => {}) },
      { keys: ["a", "/"], label: "ask", action: () => askRef.current?.focus() },
      { keys: ["c"], label: "camera", action: toggleCamera },
      { keys: ["p"], label: "pause", action: togglePause },
      { keys: ["m"], label: "mute", action: () => settings && updateSettings({ announcements_muted: !settings.announcements_muted }) },
      { keys: ["t"], label: "theme", action: () => updateSettings({ theme: theme === "dark" ? "light" : "dark" }) },
      { keys: [","], label: "settings", action: () => setShowSettings(true) },
      { keys: ["?"], label: "help", action: () => setShowShortcuts(true) },
    ],
    !showSettings && !showShortcuts,
  );

  return (
    <div className="min-h-screen bg-bg text-ink">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-xl focus:bg-sage focus:px-4 focus:py-3 focus:font-bold focus:text-sage-contrast"
      >
        Skip to main controls
      </a>
      <Header
        camera={camera}
        connected={live.connected || !live.everConnected}
        theme={theme}
        onToggleTheme={() => updateSettings({ theme: theme === "dark" ? "light" : "dark" })}
        onShortcuts={() => setShowShortcuts(true)}
        onSettings={() => setShowSettings(true)}
      />
      <SafetyNotice />

      {notice && (
        <div className="mx-auto mt-4 max-w-[92rem] px-4 sm:px-8" role="alert">
          <p className="rounded-xl border border-danger/40 bg-danger-soft px-4 py-3 font-bold text-danger">{notice}</p>
        </div>
      )}

      <main id="main" tabIndex={-1} className="mx-auto grid max-w-[92rem] gap-5 px-4 py-6 sm:px-8 lg:grid-cols-12">
        <div className="space-y-5 lg:col-span-7 xl:col-span-8">
          <CameraPanel
            camera={camera}
            objects={live.objects}
            frame={live.frame}
            cameras={cameras}
            selectedIndex={settings?.camera_index ?? 0}
            onSelect={(i) => (cameraActive ? startCamera(i) : updateSettings({ camera_index: i }))}
            onStart={() => startCamera()}
            onPause={() => api.pauseCamera()}
            onResume={() => api.resumeCamera()}
            onStop={stopCamera}
            busy={camBusy}
            showBoxes={showBoxes}
            onToggleBoxes={() => setShowBoxes((v) => !v)}
          />
          <ScenePanel
            scene={live.scene}
            progress={live.sceneProgress}
            vlm={live.status?.vlm}
            cameraActive={cameraActive}
            pending={describing}
            error={sceneError}
            onDescribe={describe}
          />
          <AskPanel ref={askRef} turns={turns} pending={asking} cameraActive={cameraActive} onAsk={ask} />
        </div>

        <div className="space-y-5 lg:col-span-5 xl:col-span-4">
          <ObjectsPanel objects={live.objects} active={camState === "running" || camState === "paused"} onSummary={() => api.summary()} />
          {settings && (
            <AudioPanel
              settings={settings}
              speech={live.speech}
              voices={voices}
              onChange={updateSettings}
              onStop={stopSpeech}
              onTest={() => api.testSpeech()}
            />
          )}
          <ActivityFeed events={live.activity} />
          <StatusPanel status={live.status} connected={live.connected} onRetryVlm={() => api.startVlm().catch(() => {})} />
        </div>
      </main>

      <footer className="mx-auto max-w-[92rem] px-4 pb-10 text-sm text-muted sm:px-8">
        <p>
          AuraSight supports, but does not replace, a white cane, guide dog or other mobility aid. It cannot see drop-offs,
          steps or hazards reliably, nor anything outside the camera's view. All processing happens on this computer.
          Press <kbd className="font-mono">?</kbd> for keyboard shortcuts.
        </p>
      </footer>

      {settings && (
        <SettingsDialog open={showSettings} onClose={() => setShowSettings(false)} settings={settings} onChange={updateSettings} />
      )}
      <ShortcutsDialog open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <LiveRegions latest={live.activity[0]} alert={live.alert} enabled={settings?.screen_reader_announcements ?? true} />
    </div>
  );
}
