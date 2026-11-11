"""VLM adapter tests.

* Protocol tests run the real HTTP clients against a tiny local stand-in server
  that speaks the llama-server / Ollama wire format. They verify request shape,
  parsing, error handling and the analyzer's single-flight logic. They do NOT
  test model quality.
* ``test_real_local_vlm`` exercises the actual SmolVLM model and only runs when a
  llama-server is reachable on 127.0.0.1:8090 (otherwise it is skipped).
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from backend.config.settings import Settings
from backend.vision.analyzer import AnalyzerBusy, SceneAnalyzer, clean_text
from backend.vision.backends import LlamaCppBackend, OllamaBackend, VLMUnavailable


class StandIn:
    def __init__(self, reply="A desk with a laptop is ahead. A chair is on the right.", health=200, delay=0.0):
        self.reply, self.health, self.delay = reply, health, delay
        self.replies: list[str] = []  # queued replies; then falls back to self.reply
        self.requests: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._send(outer.health, {"status": "ok" if outer.health == 200 else "loading"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": "moondream:latest"}]})
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(n))
                outer.requests.append({"path": self.path, "body": body})
                time.sleep(outer.delay)
                reply = outer.replies.pop(0) if outer.replies else outer.reply
                if self.path == "/v1/chat/completions" and body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for tok in reply.split(" "):
                        chunk = {"choices": [{"delta": {"content": tok + " "}}]}
                        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                elif self.path == "/v1/chat/completions":
                    self._send(200, {"choices": [{"message": {"role": "assistant", "content": reply}}]})
                elif self.path == "/api/generate" and body.get("stream"):
                    self.send_response(200)
                    self.end_headers()
                    for tok in reply.split(" "):
                        self.wfile.write((json.dumps({"response": tok + " ", "done": False}) + "\n").encode())
                    self.wfile.write((json.dumps({"response": "", "done": True}) + "\n").encode())
                elif self.path == "/api/generate":
                    self._send(200, {"response": reply})
                else:
                    self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def standin():
    s = StandIn()
    yield s
    s.close()


FRAME = np.full((480, 640, 3), 90, np.uint8)


def test_llamacpp_request_format(standin):
    b = LlamaCppBackend(standin.url)
    assert b.health() == (True, "ready")
    out = b.generate(b"\xff\xd8jpegbytes", "Describe", 64, 5)
    assert out.startswith("A desk")
    req = standin.requests[0]
    assert req["path"] == "/v1/chat/completions"
    parts = req["body"]["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == b"\xff\xd8jpegbytes"
    assert parts[1] == {"type": "text", "text": "Describe"}
    assert req["body"]["max_tokens"] == 64


def test_ollama_request_format(standin):
    b = OllamaBackend(standin.url, "moondream")
    assert b.health()[0]
    assert b.generate(b"img", "Describe", 32, 5).startswith("A desk")
    body = standin.requests[0]["body"]
    assert body["model"] == "moondream" and body["images"] == [base64.b64encode(b"img").decode()]
    assert body["options"]["num_predict"] == 32
    assert OllamaBackend(standin.url, "llava").health()[0] is False


def test_loading_and_unreachable_states():
    s = StandIn(health=503)
    try:
        assert LlamaCppBackend(s.url).health() == (False, "model loading")
    finally:
        s.close()
    ok, msg = LlamaCppBackend("http://127.0.0.1:9").health()
    assert not ok and "not running" in msg
    with pytest.raises(VLMUnavailable):
        LlamaCppBackend("http://127.0.0.1:9").generate(b"x", "p", 8, 2)


def test_analyzer_describe_and_single_flight():
    s = StandIn(delay=0.5)
    try:
        settings = Settings(vlm_backend="llamacpp", vlm_url=s.url, vlm_autostart=False)
        an = SceneAnalyzer(lambda: settings, None, None, None, backend=LlamaCppBackend(s.url))
        assert an.ensure_ready(5)
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", an.describe(FRAME)))
        t.start()
        time.sleep(0.15)
        with pytest.raises(AnalyzerBusy):
            an.describe(FRAME)
        t.join(5)
        assert result["r"].text == "A desk with a laptop is ahead. A chair is on the right."
        assert an.status()["state"] == "ready" and an.last_latency_ms >= 400
        # requests[0] is the warm-up; the real request is streamed with repetition penalties
        req = s.requests[-1]["body"]
        assert req["stream"] is True and req["repeat_penalty"] > 1
        assert req["max_tokens"] == settings.vlm_max_tokens
        assert req["messages"][0]["content"][1]["text"] == (
            "Describe this image for a blind person in two or three short sentences.")
        # image is resized to vlm_image_size before sending
        url = req["messages"][0]["content"][0]["image_url"]["url"]
        import cv2
        img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) == settings.vlm_image_size
    finally:
        s.close()


def test_analyzer_unavailable_raises():
    settings = Settings(vlm_autostart=False)
    an = SceneAnalyzer(lambda: settings, None, None, None, backend=LlamaCppBackend("http://127.0.0.1:9"))
    with pytest.raises(VLMUnavailable):
        an.describe(FRAME)
    assert an.state == "unavailable"


def test_clean_text():
    assert clean_text("The image shows a kitchen. A kettle is on the left.") == (
        "I can see a kitchen. A kettle is on the left.")
    # full sentences followed by a token-limit cut: the fragment is dropped, not spoken
    assert clean_text("A desk is ahead. A chair is on the right. Behind it there") == (
        "A desk is ahead. A chair is on the right.")
    assert clean_text("In this image, a desk. A desk. A chair.") == "A desk. A chair."
    assert clean_text("One. Two. Three. Four.") == "One. Two. Three."
    assert clean_text("A room with a table and a lamp next to the") .endswith(".")
    assert clean_text("<end_of_utterance>A cat sits.") == "A cat sits."


REAL_SCENES = [  # image, words of which at least one must appear in the description
    ("person_living_room.jpg", ("man", "person", "sitting", "chair")),
    ("window_door_living_room.jpg", ("window", "door", "room")),
    ("tv_living_room_1998.jpg", ("television", "tv")),
    ("table_objects.jpg", ("table", "plate", "flower", "fruit", "vase")),
    ("empty_room.jpg", ("empty", "wall", "room")),
]


def _real_backend():
    b = LlamaCppBackend("http://127.0.0.1:8090")
    if not b.health()[0]:
        pytest.skip("local llama-server not running on :8090")
    settings = Settings(vlm_autostart=False)
    return SceneAnalyzer(lambda: settings, None, None, None, backend=b)


@pytest.mark.parametrize("image,words", REAL_SCENES)
def test_real_local_vlm_describes_scenes(image, words):
    """Actual SmolVLM: descriptions are multi-sentence, relevant, and never the 'Left.' failure."""
    import cv2
    from pathlib import Path as _P

    an = _real_backend()
    img = cv2.imread(str(_P(__file__).parent / "fixtures" / "scenes" / image))
    res = an.describe(img)
    assert len(res.text.split()) >= 10, res.raw
    assert any(w in res.text.lower() for w in words), res.text
    assert "image" not in res.text.lower().split()[:3]  # filler like 'In this image' removed


def test_real_local_vlm_answers_questions():
    import cv2
    from pathlib import Path as _P

    an = _real_backend()
    img = cv2.imread(str(_P(__file__).parent / "fixtures" / "scenes" / "table_objects.jpg"))
    res = an.ask(img, "What objects are on the table?")
    assert any(w in res.text.lower() for w in ("plate", "flower", "fruit", "book", "vase")), res.text



@pytest.mark.parametrize("raw,expected", [
    # Real SmolVLM phrasing -> the user's frame of reference (webcam is not mirrored)
    ("There is some text on the wall at right side of the picture.", "There is some text on the wall on your right."),
    ("On right side corner we have an empty chair.", "On your right we have an empty chair."),
    ("There is a door on the right side of the image and also on the left side of the image.",
     "There is a door on your right and also on your left."),
    ("In the center of the picture we can see a table.", "Straight ahead I can see a table."),
    ("In the background, we can see trees.", "Further away, I can see trees."),
    ("This is an empty room.", "This looks like an empty room."),
    # relational positions are NOT rewritten
    ("The chair is to the left of the table.", "The chair is to the left of the table."),
    ("On the left side of the wall there is a shelf.", "On the left side of the wall there is a shelf."),
    # repetition loops collapsed, meta-commentary dropped
    ("In this image we can see a television, a clock, a bottle, a bottle, a bottle, a cup.",
     "I can see a television, a clock, a bottle, a cup."),
    ("No chair. The focus should be on the empty room.", "No chair."),
])
def test_clean_text_rewrites(raw, expected):
    assert clean_text(raw) == expected


def test_degenerate_answer_is_retried_and_never_spoken():
    """Regression for the 'Left.' bug: a one-word answer is not spoken; the
    analyzer retries with the fallback prompt and speaks that instead."""
    s = StandIn()
    try:
        settings = Settings(vlm_autostart=False)
        an = SceneAnalyzer(lambda: settings, None, None, None, backend=LlamaCppBackend(s.url))
        assert an.ensure_ready(5)
        s.replies = ["Left.", "This is a living room. A man is sitting on a chair on the right side of the image."]
        spoken = []
        res = an.describe(FRAME, ["person"], on_sentence=spoken.append)
        assert res.retried
        assert res.text == "This looks like a living room. A man is sitting on a chair on your right."
        assert spoken == ["This looks like a living room.", "A man is sitting on a chair on your right."]
        prompts = [r["body"]["messages"][0]["content"][1]["text"] for r in s.requests[-2:]]
        assert prompts[0].startswith("An object detector found: person.")
        assert prompts[1] == "Describe this image in two or three sentences."
        assert an.debug()["raw"].startswith("This is a living room")
    finally:
        s.close()


def test_sentences_are_released_while_streaming():
    from backend.vision.analyzer import SentenceStream

    got = []
    st = SentenceStream(got.append, max_sentences=3, min_words=5)
    for tok in "The picture shows a quiet kitchen. A kettle sits on the left side of the image. Behind".split(" "):
        st.feed(tok + " ")
    assert got == ["I can see a quiet kitchen.", "A kettle sits on your left."]
    st.finish()
    assert len(got) == 2  # the cut-off fragment "Behind" is dropped


def test_short_answer_is_held_back_until_final():
    from backend.vision.analyzer import SentenceStream

    got = []
    st = SentenceStream(got.append, max_sentences=3, min_words=5)
    st.feed("Left. ")
    assert got == []
    st.finish(final=False)
    assert got == []
    st.finish(final=True)
    assert got == ["Left."]


def test_ollama_streaming(standin):
    b = OllamaBackend(standin.url, "moondream")
    parts = []
    out = b.generate(b"img", "Describe", 32, 5, {"repeat_penalty": 1.2}, on_delta=parts.append)
    assert out.startswith("A desk") and len(parts) > 3
    assert standin.requests[-1]["body"]["options"]["repeat_penalty"] == 1.2
