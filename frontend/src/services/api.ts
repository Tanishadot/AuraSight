import type { AnswerResult, CameraInfo, CameraStatus, SceneResult, Settings, Voice } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the AuraSight service on this computer. Is the backend running?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  settings: () => request<Settings>("/api/settings"),
  updateSettings: (patch: Partial<Settings>) =>
    request<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),
  cameras: () => request<CameraInfo[]>("/api/cameras"),
  startCamera: (index?: number) => post<CameraStatus>("/api/camera/start", index === undefined ? {} : { index }),
  pauseCamera: () => post<CameraStatus>("/api/camera/pause"),
  resumeCamera: () => post<CameraStatus>("/api/camera/resume"),
  stopCamera: () => post<CameraStatus>("/api/camera/stop"),
  describe: () => post<SceneResult>("/api/scene/describe", { speak: true }),
  replayScene: () => post<{ description: string; speech: SceneResult["speech"] }>("/api/scene/replay"),
  ask: (question: string) => post<AnswerResult>("/api/ask", { question, speak: true }),
  summary: () => post<{ text: string }>("/api/summary"),
  stopSpeech: () => post<{ stopped: boolean }>("/api/speech/stop"),
  testSpeech: () => post<{ queued: boolean }>("/api/speech/test"),
  voices: () => request<Voice[]>("/api/speech/voices"),
  startVlm: () => post("/api/vlm/start"),
  hardware: () => request<Record<string, unknown>>("/api/hardware"),
};
