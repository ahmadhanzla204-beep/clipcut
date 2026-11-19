"""Check the text that goes with a clip: how the AI is asked for it, how tags are cleaned, and the clip editor's tags."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main(data):
    os.environ.update(PEAKCUT_DATA=data, DATABASE_URL="", PEAKCUT_ACCOUNTS="0")
    from studio import auth, brain, ctx, db, jobs, posting, server

    db.init()
    try:
        # ---- how the AI is asked
        for name, text in (("the video prompt", brain.VIDEO_RULES), ("the live and hand-clip prompt", brain.RULES)):
            check(f"{name} has no leftover marker and still fills in", "@@POST@@" not in text and "{transcript}" in text
                  and len(text.format(min_len=15, max_len=60, transcript="x")) > 500)
            check(f"{name} asks for a title people search for", "at most 60 characters" in text and "search" in text)
            check(f"{name} asks for ten one-word tags", "10 tags" in text and "ONE word" in text)
            check(f"{name} asks for the language the video is spoken in", "language the video is spoken in" in text
                  and "Roman letters" in text)

        # ---- the language of the words
        hindi = [{"w": "\u0905\u092c"}, {"w": "\u092c\u093e\u0924"}, {"w": "\u0915\u0930\u0924\u0947"}, {"w": "GTA"}]
        urdu = [{"w": "\u06a9\u06cc\u0627"}, {"w": "\u06c1\u06d2"}, {"w": "\u0622\u067e"}]
        check("Hindi words are recognised by their script", brain.spoken_language(hindi) == "Hindi")
        check("Urdu words are recognised by their script", brain.spoken_language(urdu) == "Urdu")
        check("English words give no language note (the AI follows the words)", brain.spoken_language([{"w": "hello"}, {"w": "world"}]) == ""
              and brain.spoken_language([]) == "")
        note = brain.LANGUAGE_NOTE.format(lang="Hindi")
        check("the note asks for Roman letters, not plain English and not Hindi script", "Roman letters" in note and "plain English" in note)

        # ---- cleaning tags
        got = brain.clean_tags(["#GTA 6", "gta6", "GTA6", "vice city!", 5, None, {"a": 1}, "shorts"])
        check("tags lose the # and spaces, repeats and non-text are dropped", got == ["GTA6", "vicecity", "5", "shorts"])
        check("tags keep other scripts", brain.clean_tags(["शॉर्ट्स"]) == ["शॉर्ट्स"])
        check("tags are limited in number and length", len(brain.clean_tags([f"t{i}" for i in range(40)])) == 15
              and len(brain.clean_tags(["x" * 100])[0]) == 30)
        check("something that is not a list gives no tags", brain.clean_tags("a, b") == [] and brain.clean_tags(None) == [])

        # ---- what comes out of the AI's answer
        tidy = brain._tidy({"start": 0, "end": 1, "score": 70, "hook": "x", "title": "T" * 120, "description": "d" * 700,
                            "hashtags": [f"tag{i}" for i in range(12)] + ["#gta 6"]}, 0.0, 30.0)
        check("a description may be a few lines long, but not endless", 400 < len(tidy["description"]) <= 600)
        check("ten or more tags from the AI are kept", len(tidy["hashtags"]) >= 12 and tidy["hashtags"][0] == "tag0")

        # ---- the text that is posted
        post = posting._caption({"description": "A short story.", "hashtags": ["GTA6", "shorts"]}, {})
        check("the post text carries the tags after the description", post == "A short story.\n\n#GTA6 #shorts")

        # ---- the clip editor
        user = auth.local_user().id
        source = {"type": "file", "name": "Metadata test"}
        clip = {"id": "clip1", "start": 0.0, "end": 10.0, "status": "done", "hook": "H", "title": "A clip",
                "description": "", "hashtags": ["old"], "score": 80}
        with ctx.as_user(user), patch.object(jobs._q, "put") as queue:
            job = jobs.create(source, {}, enqueue=False)
            job.update(status="done", source_path=str(Path(data) / "source.mp4"), width=1920, height=1080, duration=10.0, clips=[clip])
            jobs.save(job)
            def save(changes):      # a save is refused while the clip is still being made, so it is made again first
                job["clips"][0]["status"] = "done"
                return server.rerender(job["id"], clip["id"], changes)
            res = save({"hashtags": "GTA 6, #vice city, shorts, shorts"})
            check("the clip editor saves tags typed with commas", res["clips"][0]["hashtags"] == ["GTA6", "vicecity", "shorts"])
            check("saving tags queues the clip", queue.called)
            res = save({"hashtags": ["a", "b"]})
            check("the clip editor also takes a list of tags", res["clips"][0]["hashtags"] == ["a", "b"])
            res = save({"title": "New title"})
            check("a save that does not mention tags leaves them alone", res["clips"][0]["hashtags"] == ["a", "b"])
            res = save({"hashtags": ""})
            check("empty tags clear them", res["clips"][0]["hashtags"] == [])
            save({"hashtags": "kept, saved"})
            saved = json.loads((Path(job["dir"]) / "job.json").read_text(encoding="utf-8"))["clips"][0]
            check("the tags are saved with the clip", saved["hashtags"] == ["kept", "saved"])
    finally:
        db.engine.dispose()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pc-meta-") as data:
        main(data)
