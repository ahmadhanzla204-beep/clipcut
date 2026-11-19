"""The AI layer, with the network replaced: every kind of service is asked correctly and its answer is read
correctly. Nothing leaves this machine and no key is needed."""
import json
import os
import re
import sys
import tempfile

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-ai-")
os.environ["DATABASE_URL"] = ""
os.environ.update(PEAKCUT_AI_PROVIDER="gemini", PEAKCUT_AI_MODELS="model-a, model-b", PEAKCUT_AI_KEY="test-key-123456", PEAKCUT_AI_BASE_URL="")
sys.path.insert(0, ".")
from studio import ai, auth, brain, config, ctx, db, usage  # noqa: E402

db.init()
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    ok or fails.append(name)


class Reply:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


CLIP = {"start": 1, "end": 20, "score": 80, "hook": "WOW", "title": "t", "description": "d", "hashtags": ["a"], "reason": "r"}
seen = []
story_whole = [True]
fix_calls = [0]


def fake_post(url, json=None, headers=None, timeout=None, **_k):  # noqa: A002
    seen.append((url, json, headers or {}))
    if url.endswith(":generateContent"):
        answer = [CLIP]
        text = json["contents"][0]["parts"][0]["text"]
        if "You check several short clips" in text:      # the "is each story whole?" check: one call, many clips
            given = [(int(i), float(a), float(b)) for i, a, b in re.findall(r'<clip id="(\d+)" from="([\d.]+)" to="([\d.]+)"', text)]
            if story_whole[0] == "fix":                  # first says "cut off, it should end 10 s later", then agrees with the longer clip
                fix_calls[0] += 1
                answer = [{"id": i, "complete": fix_calls[0] > 1, "start": a, "end": b + 10, "problem": "ends too early"} for i, a, b in given]
            elif story_whole[0] == "mixed":              # clip 0 whole, clip 1 cut off for good, clip 2 not answered at all
                answer = [{"id": i, "complete": i == 0, "start": a, "end": b, "problem": "cut short"} for i, a, b in given if i != 2]
                answer.append({"id": 99, "complete": False, "start": 1, "end": 20, "problem": "an id that is not a clip"})
            else:                                        # a clip that is not whole repeats its own times: it cannot be fixed
                answer = [{"id": i, "complete": story_whole[0], "start": a, "end": b, "problem": "cut short"} for i, a, b in given]
        return Reply(200, {"candidates": [{"content": {"parts": [{"text": globals()["json_text"](answer)}]}}],
                           "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 130}})
    if url.endswith("/chat/completions"):
        if "temperature" in json:                      # a model that refuses an extra setting: asked again without it
            return Reply(400, {"error": {"message": "temperature is not supported"}})
        usage_part = {"prompt_tokens": 90, "completion_tokens": 20}
        if "openrouter.ai" in url:
            usage_part["cost"] = 0.0123                  # OpenRouter says what the call really cost
        answer = [CLIP]
        if "pictures taken from one stretch" in str(json):    # the model that looks at pictures
            answer = [{"start": 3, "end": 9, "what": "a big splash", "strength": 80},
                      {"start": 500, "end": 600, "what": "far outside the pictures", "strength": 90},
                      {"start": 5, "end": 4, "what": "backwards", "strength": 50}]
        return Reply(200, {"choices": [{"message": {"content": "```json\n" + globals()["json_text"]({"items": answer}) + "\n```"}}],
                           "usage": usage_part})
    if url.endswith("/messages"):
        return Reply(200, {"content": [{"type": "text", "text": "Here you go: " + globals()["json_text"]({"items": [CLIP]})}],
                           "usage": {"input_tokens": 70, "output_tokens": 10}})
    return Reply(404, {})


def json_text(x):
    return json.dumps(x)


ai.httpx.post = fake_post
ai.httpx.get = lambda url, **_k: Reply(200, {"data": [{"id": "model-a", "pricing": {"prompt": "0.000002", "completion": "0.00001"}}]})  # a price list


def use(provider, base=""):
    config.AI_PROVIDER, config.AI_BASE_URL = provider, base
    seen.clear()


use("gemini")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("gemini: answer and tokens", out == [CLIP] and (sent, wrote) == (100, 30), str((out, sent, wrote)))
check("gemini: key in the header, schema passed as is", seen[0][2].get("x-goog-api-key") == "test-key-123456"
      and seen[0][1]["generationConfig"]["responseSchema"] is brain.SCHEMA and seen[0][0].endswith("/model-a:generateContent"))

for name, host in (("openai", "api.openai.com"), ("xai", "api.x.ai"), ("grok", "api.x.ai"), ("groq", "api.groq.com"), ("openrouter", "openrouter.ai")):
    use(name)
    out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
    check(f"{name}: answer unwrapped from a code fence, tokens read", out == [CLIP] and (sent, wrote) == (90, 20), str((out, sent, wrote)))
    check(f"{name}: right address, key as a bearer token", host in seen[0][0] and seen[0][2].get("Authorization") == "Bearer test-key-123456")
check("a refused extra setting is dropped and asked again", len(seen) == 2 and "temperature" not in seen[1][1])
check("the answer length is limited, so little money is held back", seen[1][1].get("max_tokens") == 8000, str(seen[1][1].get("max_tokens")))
check("the schema is told in the prompt for chat models", '"type": "array"' in seen[1][1]["messages"][0]["content"])
check("openrouter: asked to report the cost, and it is read", seen[1][1].get("usage") == {"include": True} and ai.last_cost() == 0.0123, str(ai.last_cost()))
use("openai")
ai.call("model-a", "hello", brain.SCHEMA)
check("another service: no cost asked for, none reported", "usage" not in seen[-1][1] and ai.last_cost() is None)

check("an answer with extra text after the JSON is still read", ai._parse('[{"a": 1}]\n[{"a": 2}]') == [{"a": 1}] and ai._parse('Here: {"items": [3]} thanks') == [3])
for name, mark, nxt in (("Hindi", "\u0964", "\u0905\u092c"), ("Urdu", "\u06d4", "\u0627\u0628"), ("Urdu question", "\u061f", "\u0627\u0628")):
    ws = [{"w": "a", "s": 1.0, "e": 1.2}, {"w": "one" + mark, "s": 9.6, "e": 10.0}, {"w": "b", "s": 10.0, "e": 10.2},
          {"w": "two" + mark, "s": 19.6, "e": 20.0}, {"w": nxt, "s": 20.0, "e": 20.2}, {"w": "c", "s": 20.2, "e": 21.0}, {"w": "d", "s": 21.0, "e": 22.0}, {"w": "e", "s": 22.0, "e": 23.0}, {"w": "f", "s": 23.0, "e": 24.0}]
    got = brain._snap({"start": 1, "end": 20.4}, [{"s": 1.0}], ws, 15, 60, 100, sentence=True)[1]
    check(f"{name}: a clip ends where the sentence ends, not in the middle of the next one", got <= 20.0, str(got))
ws = [{"w": "a", "s": 1.0, "e": 1.2}, {"w": "one.", "s": 9.6, "e": 10.0}, {"w": "b", "s": 10.0, "e": 10.2},
      {"w": "end.", "s": 19.6, "e": 20.0}, {"w": "Now", "s": 20.0, "e": 20.2}, {"w": "c", "s": 20.2, "e": 21.0}, {"w": "d", "s": 21.0, "e": 22.0}, {"w": "e", "s": 22.0, "e": 23.0}, {"w": "f", "s": 23.0, "e": 24.0}]
check("a clip does not run into the next sentence's first word",
      brain._snap({"start": 1, "end": 20.4}, [{"s": 1.0}], ws, 15, 60, 100, sentence=True)[1] <= 20.0)

use("anthropic")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("anthropic: answer found inside a sentence, tokens read", out == [CLIP] and (sent, wrote) == (70, 10), str((out, sent, wrote)))
check("anthropic: key and version headers", seen[0][2].get("x-api-key") == "test-key-123456" and "anthropic-version" in seen[0][2])

use("custom", "http://127.0.0.1:1234/v1")
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("custom: uses the address from settings", seen[0][0] == "http://127.0.0.1:1234/v1/chat/completions" and out == [CLIP])

use("custom")
check("custom without an address is not ready", not ai.ready())
use("nonsense")
check("an unknown provider is not ready", not ai.ready())
use("ollama")
os.environ["PEAKCUT_AI_KEY"] = ""
config._kept.clear()
check("ollama needs no key", ai.ready() and ai.label().startswith("Ollama"))
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("ollama: no key header is sent", "Authorization" not in seen[0][2] and out == [CLIP])
use("gemini")
check("gemini without a key is not ready", not ai.ready())
os.environ["PEAKCUT_AI_KEY"] = "test-key-123456"
config._kept.clear()

# ---- the whole path: a call is made for a user and shows up in their usage, and only theirs
u1, _ = auth._create("ai-one@example.com", "One", password_hash="x")
u2, _ = auth._create("ai-two@example.com", "Two", password_hash="x")
words = [{"w": f"word{i}" + ("." if i % 10 == 9 else ""), "s": i * 0.5, "e": i * 0.5 + 0.4} for i in range(120)]
with ctx.as_user(u1.id):
    usage.set_context(None, "A test video")
    import numpy as np
    clips, info = brain.find_clips(words, np.zeros(60), 60.0, title="T", uploader="U")
    check("clips come back through the whole path", len(clips) == 1 and clips[0]["hook"] == "WOW" and info["model"] == "model-a", str((clips, info)))
    s = usage.summary({})
    check("the calls are in this user's usage (finding the clip, checking it)", s["calls_total"] == 2 and s["tokens_total"] == 260 and s["recent"][0]["where"] == "A test video", str(s))
    check("usage says which AI is set", s["ready"] and s["provider"] == "Gemini" and s["model"] == "model-a")
    story_whole[0] = False
    clips2, info2 = brain.find_clips(words, np.zeros(60), 60.0, title="T", uploader="U")
    check("a clip whose story is not finished is left out", clips2 == [] and "not finished" in info2["note"], str((clips2, info2)))
    story_whole[0], fix_calls[0] = "fix", 0
    clips3, info3 = brain.find_clips(words, np.zeros(60), 60.0, title="T", uploader="U")
    check("a clip that is cut off is stretched and kept, not thrown away", len(clips3) == 1 and clips3[0]["end"] > 25 and info3["note"] == "", str((clips3, info3)))
    story_whole[0] = True
    # the story check is ONE call for all the clips
    three = [{"start": 25.0 * k, "end": 19.9 + 25 * k - (10 if k == 2 else 0), "score": 70 + k} for k in range(3)]
    all_lines = brain.make_lines(words)
    def checks():
        return sum(1 for x in seen if "You check several short clips" in x[1]["contents"][0]["parts"][0]["text"])
    seen.clear()
    kept, dropped, _t = brain._check_clips(three, all_lines, words, 5, 90, 60.0, lambda *_a: None)
    check("three whole clips are checked in one call", checks() == 1 and len(kept) == 3 and dropped == 0, str((checks(), len(kept), dropped)))
    story_whole[0] = "mixed"
    seen.clear()
    kept, dropped, _t = brain._check_clips(three, all_lines, words, 5, 90, 60.0, lambda *_a: None)
    check("a clip cut off for good is left out; one the AI did not answer for is kept; still one call",
          checks() == 1 and len(kept) == 2 and dropped == 1 and [k["score"] for k in kept] == [70, 72], str((checks(), kept, dropped)))
    story_whole[0] = True
    # per-video usage: model, tokens, cost (real, estimated or unknown), and removing one video's record
    v = usage.by_video(u1.id)
    check("the calls are grouped under the video's title, with the model", len(v) == 1 and v[0]["title"] == "A test video"
          and len(v[0]["calls"]) >= 2 and v[0]["calls"][0]["model"] == "model-a", str(v))
    check("a service that does not report cost leaves it unknown", v[0]["cost"] is None and usage.cost_total(u1.id)["cost"] is None)
    use("openrouter")
    usage.set_context(None, "Second video", "job-2")
    brain.ask_ai("x", purpose="p")
    usage.record(100, 50, "p", "model-a", None)           # no reported cost: estimated from the price list
    usage.record(100, 50, "p", "model-unlisted", None)    # not in the price list: stays unknown
    v2 = [x for x in usage.by_video(u1.id) if x["job_id"] == "job-2"][0]
    cs = [c["cost"] for c in v2["calls"]]
    check("a reported cost is kept as real", cs[0] == 0.0123 and not v2["calls"][0]["est"], str(v2))
    check("a missing cost is estimated and marked", abs(cs[1] - 0.0007) < 1e-9 and v2["calls"][1]["est"], str(v2))
    check("an unpriced model stays unknown", cs[2] is None, str(v2))
    check("the video total adds them up and is marked as partly estimated", abs(v2["cost"] - 0.013) < 1e-9 and v2["est"], str(v2))
    check("deleting removes that video's records only", usage.delete_video("job-2") == 3 and len(usage.by_video(u1.id)) == 1)
    check("deleting by title (older records) works", usage.delete_video("", "A test video") >= 2 and usage.by_video(u1.id) == [])
    use("gemini")
with ctx.as_user(u2.id):
    check("another user sees none of it", usage.summary({})["calls_total"] == 0)

# ---- a separate model for the last decisions
check("without a final model, final calls use the same models", ai.models("final") == ai.models() == ["model-a", "model-b"])
ai.save("gemini", "model-scan", "", "test-key-123456", "model-final")
check("scan and final models are kept apart", ai.models() == ["model-scan"] and ai.models("final") == ["model-final"], str(ai.public()))
ai.save("gemini", "model-scan2", "", None)
check("saving without a final list keeps the one saved", ai.models("final") == ["model-final"])
ai.save("gemini", "model-scan2", "", None, [])
check("an empty final list goes back to the same models", ai.models("final") == ["model-scan2"])
seen.clear()
brain.ask_ai("x", purpose="t", level="scan")
ai.save("gemini", "model-scan", "", None, "model-final")
brain.ask_ai("x", purpose="t", level="final")
check("the final call goes to the final model", seen[-1][0].endswith("/model-final:generateContent"), str(seen[-1][0]))

# ---- a model that looks at pictures of the video
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from studio import media  # noqa: E402

check("no video model chosen: none is used", ai.models("video") == [])
ai.save("openrouter", "model-scan", "", "test-key-123456", None, "model-eyes")
check("the video model is kept apart from the others", ai.models("video") == ["model-eyes"] and ai.models() == ["model-scan"], str(ai.public()))
pics = [(0.0, b"\xff\xd8a"), (3.0, b"\xff\xd8b"), (6.0, b"\xff\xd8c")]
seen.clear()
notes = brain.look(pics, "T", 3.0)
sent_parts = seen[-1][1]["messages"][0]["content"]
check("a picture call may answer at more length", seen[-1][1].get("max_tokens") == 16000)
check("the pictures go to the video model, each after its time", seen[-1][1]["model"] == "model-eyes"
      and sum(1 for p in sent_parts if p.get("type") == "image_url") == 3 and any(p.get("text") == "[3 s]" for p in sent_parts))
check("moments are read, kept inside the pictures' time, bad ones dropped",
      len(notes) == 1 and notes[0]["what"] == "a big splash" and notes[0]["end"] <= 9.0, str(notes))
ai.save("openrouter", "model-scan", "", None, None, [])
check("without a video model nothing is asked", brain.look(pics, "T", 3.0) == [] and ai.models("video") == [])
ai.save("gemini", "model-scan", "", "test-key-123456", None, [])      # what is saved in the dashboard wins over use()
try:
    ai.call("model-a", "x", brain.SCHEMA, pics)
    check("a service that cannot take pictures says so", False)
except ai.AIError:
    check("a service that cannot take pictures says so", True)
check("what is seen is put in the prompt by time", "[5-9]" in brain._seen([{"start": 5, "end": 9, "what": "x", "strength": 50},
      {"start": 100, "end": 110, "what": "y", "strength": 60}], 0, 20) and "[100-110]" not in brain._seen([{"start": 100, "end": 110, "what": "y", "strength": 60}], 0, 20))
with tempfile.TemporaryDirectory() as tmp:
    clip = Path(tmp) / "t.mp4"
    media.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x240:rate=5", str(clip)])
    fr = media.frames(clip, 2, 11, every=3, width=160)
    check("frames: one every 3 seconds, timed from the video's start, as jpeg", len(fr) == 3 and [t for t, _ in fr] == [2.0, 5.0, 8.0] and fr[0][1][:2] == b"\xff\xd8", str([t for t, _ in fr]))
    check("frames: a stretch shorter than the gap gives none", media.frames(clip, 2, 3, every=3) == [])

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil  # noqa: E402
db.engine.dispose()
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
