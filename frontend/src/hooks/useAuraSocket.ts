import { useEffect, useReducer, useRef } from "react";
import type { ActivityEvent, SceneResult, Settings, SystemStatus, TrackedObject } from "../services/types";

export interface SpeechState {
  speaking: boolean;
  text: string | null;
  kind?: string;
}

export interface LiveState {
  connected: boolean;
  everConnected: boolean;
  status: SystemStatus | null;
  settings: Settings | null;
  objects: TrackedObject[];
  frame: { width: number; height: number; inferenceMs: number; fps: number };
  activity: ActivityEvent[];
  scene: (SceneResult & { ts: number }) | null;
  /** live text while a description is being generated */
  sceneProgress: { stage: "idle" | "analysing" | "streaming"; text: string };
  speech: SpeechState;
  /** latest urgent warning, for the assertive live region */
  alert: ActivityEvent | null;
}

type Action =
  | { type: "open" }
  | { type: "close" }
  | { type: "event"; ev: any };

const MAX_ACTIVITY = 120;

const initial: LiveState = {
  connected: false,
  everConnected: false,
  status: null,
  settings: null,
  objects: [],
  frame: { width: 4, height: 3, inferenceMs: 0, fps: 0 },
  activity: [],
  scene: null,
  sceneProgress: { stage: "idle", text: "" },
  speech: { speaking: false, text: null },
  alert: null,
};

function reducer(state: LiveState, action: Action): LiveState {
  switch (action.type) {
    case "open":
      return { ...state, connected: true, everConnected: true };
    case "close":
      return { ...state, connected: false };
    case "event": {
      const ev = action.ev;
      switch (ev.type) {
        case "hello": {
          const history = (ev.history as any[]).filter((e) => e.type === "activity") as ActivityEvent[];
          const lastScene = [...(ev.history as any[])].reverse().find((e) => e.type === "scene");
          return {
            ...state,
            status: ev.status,
            settings: ev.settings,
            activity: history.slice(-MAX_ACTIVITY).reverse(),
            scene: lastScene ? { ...lastScene } : state.scene,
          };
        }
        case "status":
          return { ...state, status: ev.status };
        case "vlm_status":
          return state.status ? { ...state, status: { ...state.status, vlm: ev.vlm } } : state;
        case "settings":
          return { ...state, settings: ev.settings };
        case "detections":
          return {
            ...state,
            objects: ev.objects,
            frame: ev.width
              ? { width: ev.width, height: ev.height, inferenceMs: ev.inference_ms ?? 0, fps: ev.fps ?? 0 }
              : state.frame,
          };
        case "activity": {
          const activity = [ev as ActivityEvent, ...state.activity].slice(0, MAX_ACTIVITY);
          const alert = ev.priority === "warning" || ev.priority === "error" ? (ev as ActivityEvent) : state.alert;
          return { ...state, activity, alert };
        }
        case "scene":
          return { ...state, scene: { ...ev } };
        case "scene_progress":
          return { ...state, sceneProgress: { stage: ev.stage, text: ev.text ?? "" } };
        case "speech":
          if (ev.state === "speaking") return { ...state, speech: { speaking: true, text: ev.text, kind: ev.kind } };
          if (ev.state === "idle" || ev.state === "stopped")
            return { ...state, speech: { speaking: false, text: state.speech.text } };
          return state;
        default:
          return state;
      }
    }
  }
}

/** Live connection to the local backend with automatic reconnection. */
export function useAuraSocket(): LiveState {
  const [state, dispatch] = useReducer(reducer, initial);
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let ping: number | undefined;
    let disposed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => {
        retry.current = 0;
        dispatch({ type: "open" });
        ping = window.setInterval(() => ws?.readyState === 1 && ws.send(JSON.stringify({ type: "ping" })), 20000);
      };
      ws.onmessage = (m) => {
        try {
          dispatch({ type: "event", ev: JSON.parse(m.data) });
        } catch {
          /* ignore malformed */
        }
      };
      ws.onclose = () => {
        window.clearInterval(ping);
        dispatch({ type: "close" });
        if (disposed) return;
        const delay = Math.min(5000, 500 * 2 ** retry.current++);
        timer = window.setTimeout(connect, delay);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      window.clearInterval(ping);
      ws?.close();
    };
  }, []);

  return state;
}
