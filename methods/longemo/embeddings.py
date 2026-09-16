"""Pinned multilingual E5 event embeddings with content-addressed local caches.

Model card: https://huggingface.co/intfloat/multilingual-e5-base
Use query/passage prefixes, masked mean pooling, and unit-length vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from evaluation.io_utils import write_json
from .common import fingerprint

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


class Encoder:
    def __init__(self, model=DEFAULT_MODEL, revision=DEFAULT_REVISION, *, device="cpu", batch_size=8, threads=4):
        if not revision or revision in ("main", "master"):
            raise ValueError("embedding revision must be pinned")
        import torch
        from transformers import AutoModel, AutoTokenizer

        if batch_size < 1 or threads < 1:
            raise ValueError("embedding batch size/threads must be positive")
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model, revision=revision, use_safetensors=True, trust_remote_code=False).to(device).eval()
        self.device, self.batch_size = device, batch_size
        self.lock = threading.Lock()
        self.config = {"model": model, "revision": revision, "pooling": "masked_mean", "normalized": True,
                       "max_length": 512, "chunk_tokens": 448, "chunk_stride": 384,
                       "device": device, "dtype": "float32", "dimension": self.model.config.hidden_size}

    def chunks(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return [""]
        return [self.tokenizer.decode(ids[i:i+448], skip_special_tokens=True) for i in range(0, len(ids), 384)]

    def encode(self, texts, *, query=False):
        import numpy as np

        if not texts:
            return np.zeros((0, self.config["dimension"]), dtype="float32")
        prefix = "query: " if query else "passage: "
        vectors = []
        with self.lock, self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer([prefix + text for text in texts[i:i+self.batch_size]],
                    max_length=512, truncation=True, padding=True, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).bool()
                pooled = hidden.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype("float32")


class APIEncoder:
    """Gemini Embedding 2 through the OpenRouter embedding endpoint."""
    def __init__(self, api_key, directory, *, model="google/gemini-embedding-2",
                 base_url="https://openrouter.ai/api/v1", dimension=3072, tries=3):
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.query or parsed.username:
            raise ValueError("embedding endpoint must be HTTPS without inline credentials")
        if model != "google/gemini-embedding-2":
            raise ValueError("API task prefixes are implemented for Gemini Embedding 2 only")
        if not api_key:
            raise ValueError("embedding API key is missing")
        self.key, self.tries = api_key, tries
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = {"model": model, "base_url": base_url.rstrip("/"), "dimension": dimension,
            "normalized": True, "dtype": "float32", "chunk_characters": 6000, "chunk_stride": 5400,
            "query_task": "search result", "document_title": "Emotional event",
            "revision": "provider-stable-alias; snapshot not exposed"}

    def chunks(self, text):
        return [text[i:i+6000] for i in range(0, len(text), 5400)] or [""]

    def encode(self, texts, *, query=False):
        import numpy as np
        from urllib import request, error
        from .common import append_json, token_usage
        if not texts:
            return np.zeros((0,self.config["dimension"]),dtype="float32")
        # Cache each batch by exact text + role + model. Queries are cached too.
        result = []
        for offset in range(0,len(texts),16):
            batch = texts[offset:offset+16]
            inputs = [("task: search result | query: " if query else "title: Emotional event | text: ") + t for t in batch]
            signature = fingerprint({"config":self.config,"inputs":inputs})
            path = self.directory/(signature+".json")
            if path.exists():
                values = json.loads(path.read_text())["vectors"]
            else:
                payload = {"model":self.config["model"],"input":inputs,"dimensions":self.config["dimension"],"encoding_format":"float"}
                for attempt in range(1,self.tries+1):
                    started=time.monotonic()
                    record={"purpose":"embedding_query" if query else "embedding_documents", "request_hash":signature,
                            "attempt":attempt,"time_unix":time.time(),"model":self.config["model"]}
                    try:
                        req=request.Request(self.config["base_url"]+"/embeddings",data=json.dumps(payload).encode(),
                            headers={"Content-Type":"application/json","Authorization":"Bearer "+self.key},method="POST")
                        with request.urlopen(req,timeout=120) as response:
                            data=json.load(response)
                        record["usage"]=token_usage(data.get("usage"))
                        rows=sorted(data["data"],key=lambda row:row["index"])
                        if [r["index"] for r in rows] != list(range(len(batch))):
                            raise ValueError("embedding response indices differ from batch")
                        values=[r["embedding"] for r in rows]
                        self._validate(values,len(batch))
                        write_json(path,{"vectors":values,"model_returned":data.get("model"),"signature":signature})
                        record.update(status="ok",elapsed_seconds=time.monotonic()-started)
                        append_json(self.directory/"calls.jsonl",record)
                        break
                    except Exception as exc:
                        record.update(status="error",error_type=type(exc).__name__,elapsed_seconds=time.monotonic()-started)
                        if isinstance(exc,error.HTTPError): record["http_status"]=exc.code
                        append_json(self.directory/"calls.jsonl",record)
                        if attempt==self.tries: raise RuntimeError("embedding API failed; inspect its call ledger") from None
                        time.sleep(min(2**attempt,8))
            array=self._validate(values,len(batch))
            result.append(array/np.linalg.norm(array,axis=1,keepdims=True))
        return np.concatenate(result).astype("float32")

    def _validate(self, values, count):
        import numpy as np
        array=np.asarray(values,dtype="float32")
        if array.shape != (count,self.config["dimension"]) or not np.isfinite(array).all() or (np.linalg.norm(array,axis=1)<=0).any():
            raise ValueError("invalid embedding dimensions/values")
        return array


def event_documents(memory):
    """Embed all state/observation evidence, not only a lossy event summary."""
    people = {p["id"]: p for p in memory["entities"]}
    observations = {o["id"]: o for o in memory["observations"]}
    result = []
    for event in memory["events"]:
        refs = set(event["observation_refs"]) | {r for s in event["states"] for r in s["evidence_refs"]}
        subjects = {s["subject"] for s in event["states"]} | {observations[r]["subject"] for r in refs}
        fields = ["People: " + "; ".join(f"{people[p]['name'] or p}: {people[p]['description']}" for p in sorted(subjects)),
                  "Video times: " + str(event["spans"])]
        if not event.get("summary_stale"):
            fields += ["Event: " + event["summary"]] + [u["summary"] for u in event.get("updates", [])]
        for state in event["states"]:
            fields.append(f"{people[state['subject']]['name'] or state['subject']} feels {state['emotion']} about {state['target']}; "
                          f"manifestations: {state['intensity']}; interpretation: {state.get('appraisal')}; uncertainty: {state.get('uncertainty')}")
        fields += ["Observed: " + observations[r]["cue"] for r in sorted(refs)]
        result.append((event["id"], "\n".join(fields)))
    return result


class EventIndex:
    def __init__(self, memory, encoder, directory):
        import numpy as np

        self.encoder = encoder
        start = time.monotonic()
        documents = event_documents(memory)
        signature = fingerprint({"documents": documents, "encoder": encoder.config})
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (memory["video_id"] + "-" + signature + ".npz")
        reused = path.exists()
        if reused:
            with np.load(path, allow_pickle=False) as stored:
                self.vectors = stored["vectors"]
                self.event_ids = stored["event_ids"].tolist()
            if self.vectors.shape != (len(self.event_ids), encoder.config["dimension"]) or not np.isfinite(self.vectors).all():
                raise ValueError("invalid cached embedding matrix")
        else:
            self.event_ids, chunks = [], []
            for event_id, document in documents:
                pieces = encoder.chunks(document)
                self.event_ids.extend([event_id]*len(pieces))
                chunks.extend(pieces)
            self.vectors = encoder.encode(chunks)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, vectors=self.vectors, event_ids=np.asarray(self.event_ids, dtype=str))
            temporary.replace(path)
        self.metadata = {"signature": signature, "video_id": memory["video_id"], "events": len(documents),
                         "chunks": len(self.event_ids), "encoder": encoder.config, "cache_reused": reused,
                         "elapsed_seconds": time.monotonic()-start, "vector_bytes": self.vectors.nbytes}
        write_json(path.with_suffix(".json"), self.metadata)

    def rank(self, question):
        vector = self.encoder.encode([question], query=True)[0]
        values = self.vectors @ vector
        scores = {}
        for event_id, score in zip(self.event_ids, values):
            scores[event_id] = max(scores.get(event_id, -1.0), float(score))
        return scores


if __name__ == "__main__":
    encoder = Encoder()
    vectors = encoder.encode(["A man is relieved after his son returns safely.", "A woman angrily rejects an offer." ])
    query = encoder.encode(["Who feels less worried after a family member comes home?"], query=True)
    print(json.dumps({"encoder": encoder.config, "shape": list(vectors.shape), "cosine": (vectors @ query[0]).tolist()}))
