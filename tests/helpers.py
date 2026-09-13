"""Synthetic questions and a local HTTP service; no dataset or paid API calls."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def question(qid="Q1", granularity="clip", task=None, **changes):
    q = {
        "question_id": qid,
        "video_id": "V1",
        "source": {"from": "SYNTHETIC_SOURCE", "segments": None},
        "granularity": granularity,
        "type": task
        or ("contextual emotion" if granularity == "clip" else "emotional reasoning"),
        "question": "What emotion does the person show?",
        "answer": ["happiness"],
        "answer_details": None,
        "rubric": None,
        "subtitles": [
            {
                "id": "U1",
                "t": [10, 11],
                "speaker": "PRIVATE_SPEAKER",
                "text": "First subtitle.",
            },
            {
                "id": "U2",
                "t": [11, 12],
                "speaker": "PRIVATE_SPEAKER",
                "text": "Last subtitle.",
            },
        ],
    }
    if granularity == "episode":
        q["answer"] = "Yes"
        q["rubric"] = {
            "criterion": "Judge whether the requested result is correct and complete.",
            "scores": {
                "0": "The answer is incorrect, incomplete or contradictory.",
                "1": "The answer is correct and complete. Accept equivalent wording or an explicitly permitted alternative.",
            },
        }
    if q["type"] == "emotion transition":
        q["answer"] = {"before": ["fear", "sadness"], "after": ["peace"]}
    if q["type"] in {"emotion cause", "emotion trajectory"}:
        q["answer"] = "REFERENCE_NOT_FOR_INFERENCE"
        maximum = 4 if q["type"] == "emotion trajectory" else 3
        q["rubric"] = {
            "criterion": "Judge the requested emotional explanation.",
            "scores": {
                str(i): f"Synthetic calibration level {i}." for i in range(maximum + 1)
            },
        }
        if maximum == 4:
            q["answer_details"] = {
                "description": "Required stages in chronological order.",
                "items": [
                    {
                        "id": "S1",
                        "description": "Initially happy.",
                        "emotion": "happiness",
                        "anchor": None,
                        "intensity": None,
                    },
                    {
                        "id": "S2",
                        "description": "Later sad.",
                        "emotion": "sadness",
                        "anchor": None,
                        "intensity": None,
                    },
                ],
            }
    q.update(changes)
    return q


def prediction(q, answer, **changes):
    return {
        "question_id": q["question_id"],
        "granularity": q["granularity"],
        "prediction": answer,
        **changes,
    }


@contextmanager
def service(responder):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            answer = responder(self.path, body)
            self.send_response(200)
            if isinstance(answer, dict):
                payload = answer
            elif self.path.endswith("/responses"):
                payload = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": answer}],
                        }
                    ],
                }
            elif ":generateContent" in self.path:
                payload = {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"text": "hidden", "thought": True},
                                    {"text": answer},
                                ]
                            },
                        }
                    ]
                }
            elif self.path.endswith("/messages"):
                payload = {
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": answer},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                }
            else:
                payload = {
                    "choices": [
                        {"message": {"content": answer}, "finish_reason": "stop"}
                    ]
                }
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
