export type CameraStateName = "stopped" | "starting" | "running" | "paused" | "error";

export interface CameraStatus {
  state: CameraStateName;
  error: string | null;
  index: number | null;
  resolution: [number, number] | null;
  fps: number;
  active: boolean;
}

export interface DetectorStatus {
  state: "not_loaded" | "loading" | "ready" | "error";
  error: string | null;
  model: string;
  inference_ms: number;
}

export interface VlmStatus {
  state: "not_started" | "disabled" | "starting" | "ready" | "busy" | "unavailable" | "error";
  detail: string;
  model: string;
  backend: string | null;
  last_latency_ms: number | null;
}

export interface SpeechStatus {
  state: "not_started" | "ready" | "speaking" | "error";
  error: string | null;
  engine: string | null;
  speaking: string | null;
  queued: number;
}

export interface SystemStatus {
  camera: CameraStatus;
  detector: DetectorStatus;
  vlm: VlmStatus;
  speech: SpeechStatus;
  pipeline: { detection_fps: number; frames_processed: number; frames_skipped: number; objects: number };
}

export interface Settings {
  camera_index: number;
  camera_width: number;
  camera_height: number;
  preview_fps: number;
  preview_quality: number;
  detector_model: string;
  inference_size: 256 | 320 | 416 | 480 | 512 | 640;
  detection_fps: number;
  confidence_threshold: number;
  detector_threads: number;
  vlm_backend: "llamacpp" | "ollama" | "none";
  vlm_url: string;
  vlm_model: string;
  vlm_mmproj: string;
  vlm_autostart: boolean;
  vlm_threads: number;
  vlm_image_size: number;
  vlm_max_tokens: number;
  vlm_timeout_s: number;
  auto_describe: boolean;
  auto_describe_interval_s: number;
  speech_enabled: boolean;
  speech_rate: number;
  speech_volume: number;
  voice_id: string | null;
  announcements_muted: boolean;
  announce_new_objects: boolean;
  announce_departures: boolean;
  announce_scene_changes: boolean;
  obstacle_alerts: boolean;
  announce_cooldown_s: number;
  min_track_frames: number;
  leave_after_s: number;
  verbosity: "brief" | "detailed";
  theme: "system" | "light" | "dark";
  screen_reader_announcements: boolean;
}

export interface TrackedObject {
  track_id: number;
  label: string;
  spoken_label: string;
  confidence: number;
  box: [number, number, number, number];
  horizontal: "left" | "centre" | "right";
  direction: string;
  distance: "near" | "middle" | "far";
  in_path: boolean;
  phrase: string;
  age_s: number;
  approaching: boolean;
  /** confirmed objects are announced; tentative ones are shown as unverified only */
  status: "confirmed" | "tentative" | "rejected";
  status_reason: string;
}

export interface ActivityEvent {
  type: "activity";
  id: number;
  ts: number;
  kind: string;
  message: string;
  priority: "info" | "warning" | "error";
  sources?: string[];
  source?: string;
}

export interface SceneResult {
  description: string;
  source: "vlm" | "detector";
  latency_ms: number;
  first_sentence_ms?: number | null;
  auto?: boolean;
  objects?: string;
}

export interface AnswerResult {
  question: string;
  answer: string;
  intent: string;
  sources: string[];
  grounded: boolean;
  latency_ms: number;
  vlm_used: boolean;
}

export interface CameraInfo {
  index: number;
  name: string;
}

export interface Voice {
  id: string;
  name: string;
}
