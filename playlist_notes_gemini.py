#!/usr/bin/env python3
"""
playlist_notes_gemini.py - Turn a YouTube playlist (or single video) into detailed lecture notes
with Google's Gemini API (free tier works).

Modes (--mode):
  diagrams  (default)  Fast. Reads the transcript and also draws flowcharts for the processes,
                       flows and structures the teacher explains.
  text                 Fast. Transcript only, no diagrams.
  watch                Slow. Gemini watches the video and describes what is actually drawn on
                       screen. Uses a lot of free quota. Public videos only.

Finished parts are saved in <out>/.cache, so if you hit the free limit you can just run the same
command again later and it continues where it stopped.

Usage:
  python playlist_notes_gemini.py "https://www.youtube.com/playlist?list=XXXX"
  python playlist_notes_gemini.py "<link>" --mode watch --low-res
  python playlist_notes_gemini.py "<link>" --language Telugu --out my_notes

Your API key is read from (first match wins):
  1. the GEMINI_API_KEY environment variable
  2. .streamlit/secrets.toml   ->  GEMINI_API_KEY = "your-key"
  3. a file called gemini_key.txt next to this script (just the key on one line)
Get a free key at https://aistudio.google.com/apikey
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from google import genai
from google.genai import types as genai_types
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig

DEFAULT_MODEL = "gemini-3.8-flash"
HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "gemini_key.txt")

# ----------------------------------------------------------------------------- prompts

_RULES = """- Write notes in {language}.
- Follow the lecture in the exact order the teacher explained things.
- COMPLETENESS IS THE MOST IMPORTANT RULE. A student who never watched the lecture must be able
  to learn everything the teacher taught from your notes alone. Go through the material from
  start to finish and make sure EVERY idea, point, definition, example, number, name, command,
  formula, step, analogy, comparison, tip and warning is in your notes, including the minor ones.
- Explain each point in detail (what it is, why it matters, how it works, and the teacher's own
  example for it). Do NOT summarise, do NOT merge several points into one line, and do NOT leave
  anything out. Keep the teacher's examples and any useful wording.
- {length_hint}
- Remove filler only (greetings, "um", repetition, "subscribe", etc.).
- Fix obvious transcription mistakes in technical terms using context.
- Use Markdown: ## headings for topics, bullet points for details, **bold** for key terms,
  and code blocks / plain-text formulas where relevant.
"""

NOTES_PROMPT = (
    """You are writing detailed study notes from a lecture transcript.

Lecture title: {title}
This is part {idx} of {total} of the lecture (timestamps {start} to {end}).

Rules:
"""
    + _RULES
    + """- If the teacher mentions something shown on screen that is not spoken, note it as
  "(shown on screen: ...)" only if the transcript hints at it. Do not invent content.
- Output only the notes, with no preamble.

Transcript:
{text}
"""
)

NOTES_PROMPT_DIAGRAMS = (
    """You are writing detailed study notes from a lecture transcript.

Lecture title: {title}
This is part {idx} of {total} of the lecture (timestamps {start} to {end}).

Rules:
"""
    + _RULES
    + """- DIAGRAMS: whenever the teacher explains a process, flow, request/response sequence,
  architecture, hierarchy, lifecycle, or how components connect, add a diagram right after that
  explanation:
    * write "**Diagram:**" and then a ```mermaid code block (flowchart TD, flowchart LR or
      sequenceDiagram);
    * keep it small (3 to 10 nodes), with short labels, and put every label in double quotes,
      for example A["Client"] --> B["Server"];
    * only show things the teacher actually explained. Never invent steps;
    * add one for every process, flow or structure that is explained (usually 0 to 4 per
      part), and none if nothing in this part is diagram-worthy. The diagram is IN ADDITION to
      the written explanation, never a replacement for it.
- When the teacher compares things, add a Markdown table.
- Output only the notes, with no preamble.

Transcript:
{text}
"""
)

VISUAL_PROMPT = (
    """You are writing detailed study notes from a lecture video. You can see and hear it.

Lecture title: {title}
This is part {idx} of {total} of the lecture (this clip is {start} to {end} of the video).

Rules:
"""
    + _RULES
    + """- Whenever the teacher shows or draws a diagram, flowchart, architecture picture, graph, table,
  slide, code or board work, include it in the notes at the point it appears:
    * write "**Diagram:**" followed by a clear description of what is drawn (boxes, arrows,
      labels, what connects to what) and what the teacher says about it;
    * if it is a flow or structure, also add a small ```mermaid code block that recreates it,
      with every label in double quotes;
    * copy any text, code or formulas shown on screen exactly.
- Do not invent anything that is not in the clip.
- Output only the notes, with no preamble.
"""
)

MODE_LABELS = {
    "diagrams": "Fast notes + diagrams",
    "text": "Fast, text only",
    "watch": "Exact: watch the video",
}


def default_chunk_minutes(mode):
    """Shorter parts keep the notes detailed; longer parts use fewer requests."""
    return 5 if mode == "watch" else 10


# ----------------------------------------------------------------------------- errors


class GeminiError(RuntimeError):
    """A problem that retrying will not fix (bad key, unknown model, ...)."""


class QuotaExhausted(RuntimeError):
    """The free limit was hit. daily=True means waiting a few minutes will not help."""

    def __init__(self, message, daily=False):
        super().__init__(message)
        self.daily = daily


def friendly_error(err):
    """A short, plain-language message for the user."""
    if isinstance(err, QuotaExhausted):
        if err.daily:
            return ("You have used up today's free Gemini requests for this model. "
                    "Your finished parts are saved. Run it again after the limit resets "
                    "(usually around midday in India), or try another model with --model / "
                    "the Advanced settings. Check ai.dev/rate-limit for your numbers.")
        return ("Gemini's free per-minute limit was hit and kept failing. Your finished parts "
                "are saved. Wait a few minutes and run it again, or lower 'parts at once'.")
    return str(err)


# ----------------------------------------------------------------------------- API key


def load_api_key():
    """Find the Gemini key without ever hard-coding it. Returns '' if none is found."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    for base in (HERE, os.getcwd()):
        path = os.path.join(base, ".streamlit", "secrets.toml")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    m = re.search(r'GEMINI_API_KEY\s*=\s*["\']([^"\']+)["\']', f.read())
                if m:
                    return m.group(1).strip()
            except OSError:
                pass
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, encoding="utf-8") as f:
                return f.readline().strip()
        except OSError:
            pass
    return ""


def save_api_key(key):
    """Remember the key in gemini_key.txt next to this script (local use only)."""
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip() + "\n")


# ----------------------------------------------------------------------------- YouTube


def get_videos(url):
    """Return a list of (video_id, title) for a playlist or single video link."""
    opts = {"quiet": True, "extract_flat": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if "entries" in info:
        return [(e["id"], e.get("title") or e["id"]) for e in info["entries"] if e]
    return [(info["id"], info.get("title") or info["id"])]


def get_transcript(video_id, proxy_url=None):
    """Return a list of {'text', 'start'} snippets, preferring manual then auto captions."""
    if proxy_url:
        api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url))
    else:
        api = YouTubeTranscriptApi()
    tlist = api.list(video_id)
    codes = [tr.language_code for tr in tlist]
    try:
        t = tlist.find_manually_created_transcript(codes)
    except Exception:
        try:
            t = tlist.find_generated_transcript(codes)
        except Exception:
            t = next(iter(tlist))
    return [{"text": s.text, "start": s.start} for s in t.fetch()]


def get_duration(video_id):
    """Video length in seconds."""
    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return int(info["duration"])


def chunk_transcript(snippets, chunk_seconds):
    """Group snippets into chunks of about chunk_seconds each: [(start, end, text)]."""
    chunks, current, chunk_start = [], [], 0.0
    for s in snippets:
        if not current:
            chunk_start = s["start"]
        current.append(s)
        if s["start"] - chunk_start >= chunk_seconds:
            chunks.append((chunk_start, s["start"], current))
            current = []
    if current:
        chunks.append((chunk_start, current[-1]["start"], current))
    return [
        (a, b, " ".join(x["text"].replace("\n", " ") for x in sn))
        for a, b, sn in chunks
    ]


def plan_chunks(video_id, mode, chunk_minutes, proxy_url=None):
    """Split a video into parts: [(start_sec, end_sec, text_or_None)]."""
    step = max(60, int(chunk_minutes * 60))
    if mode == "watch":
        duration = get_duration(video_id)
        return [(s, min(s + step, duration), None) for s in range(0, duration, step)]
    return chunk_transcript(get_transcript(video_id, proxy_url), step)


def fmt_time(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# ----------------------------------------------------------------------------- Gemini

# Notes do not need deep reasoning, and turning it down makes responses much faster. Different
# models take different settings, so try each one and fall back if the model rejects it.
_THINKING_OPTIONS = [
    lambda: genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.MINIMAL),
    lambda: genai_types.ThinkingConfig(thinking_budget=0),
    lambda: None,
]
_STATE = {"thinking": 0}


def _error_code(e):
    return getattr(e, "code", None) or getattr(e, "status_code", None)


def _retry_seconds(text):
    m = re.search(r"retry(?: in|Delay['\": ]+)\s*([\d.]+)\s*s", text, re.I)
    return float(m.group(1)) if m else None


def _generate(client, model, contents, delay, low_res=False):
    """One Gemini call with retries. Returns the text."""
    waits = [15, 45, 90, 120]
    attempt = 0
    while True:
        think = _THINKING_OPTIONS[_STATE["thinking"]]()
        config = genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=16384,
            thinking_config=think,
            media_resolution=(genai_types.MediaResolution.MEDIA_RESOLUTION_LOW
                              if low_res else None),
        )
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            out = (resp.text or "").strip()
            if not out:
                raise RuntimeError("empty response")
            if delay:
                time.sleep(delay)
            return out
        except Exception as e:
            code = _error_code(e)
            text = str(e)
            if code == 400 and _STATE["thinking"] < len(_THINKING_OPTIONS) - 1:
                _STATE["thinking"] += 1  # this model dislikes that setting; try the next one
                continue
            if code in (401, 403):
                raise GeminiError("Gemini rejected the API key. Check that it is correct and "
                                  "that you pasted the whole key.")
            if code == 404:
                raise GeminiError(f"Gemini does not know the model '{model}'. Pick another one.")
            if code == 400:
                raise GeminiError(f"Gemini refused the request: {text[:300]}")
            if code == 429 and re.search(r"PerDay|per day|daily", text, re.I):
                raise QuotaExhausted(text[:300], daily=True)
            if attempt >= len(waits):
                if code == 429:
                    raise QuotaExhausted(text[:300], daily=False)
                raise RuntimeError(f"Gemini kept failing: {text[:300]}")
            wait = _retry_seconds(text) or waits[attempt]
            wait = min(max(wait, 5), 120)
            print(f"    Gemini busy ({code or type(e).__name__}); waiting {wait:.0f}s...")
            time.sleep(wait)
            attempt += 1


def write_notes(client, model, title, idx, total, start, end, text, language, delay,
                template=NOTES_PROMPT):
    words = len(text.split())
    length_hint = (f"This part of the transcript has about {words} words. Your notes for it should "
                   f"normally be at least {int(words * 0.6)} words long. If they come out much "
                   "shorter, you have skipped something: go back and add it.")
    prompt = template.format(
        title=title, idx=idx, total=total, start=fmt_time(start), end=fmt_time(end),
        language=language, text=text, length_hint=length_hint)
    return _generate(client, model, prompt, delay)


def write_notes_visual(client, model, video_id, title, idx, total, start, end, language, delay,
                       low_res=False):
    """Let Gemini watch one clip of the YouTube video (sees diagrams and board work)."""
    minutes = max(1, int((end - start) / 60))
    length_hint = (f"This clip is about {minutes} minutes long. Write thorough notes for all of "
                   "it, from the first second to the last, and do not stop early.")
    prompt = VISUAL_PROMPT.format(
        title=title, idx=idx, total=total, start=fmt_time(start), end=fmt_time(end),
        language=language, length_hint=length_hint)
    part = genai_types.Part(
        file_data=genai_types.FileData(file_uri=f"https://www.youtube.com/watch?v={video_id}"),
        video_metadata=genai_types.VideoMetadata(
            start_offset=f"{int(start)}s", end_offset=f"{int(end)}s"),
    )
    contents = genai_types.Content(parts=[part, genai_types.Part(text=prompt)])
    return _generate(client, model, contents, delay, low_res=low_res)


# ----------------------------------------------------------------------------- jobs + cache


def safe_name(s):
    return re.sub(r"[^\w\- ]+", "", s).strip()[:80] or "video"


def cached_call(cache_dir, key, fn):
    """Return the saved result for key, or run fn() and save it."""
    path = os.path.join(cache_dir, re.sub(r"[^\w.\-]+", "_", key) + ".md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    out = fn()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return out


def build_jobs(client, model, vid, title, chunks, mode, language, delay, low_res, cache_dir):
    """One zero-argument callable per part. Finished parts are reused from cache_dir."""
    total = len(chunks)
    template = NOTES_PROMPT_DIAGRAMS if mode == "diagrams" else NOTES_PROMPT
    jobs = []
    for i, (a, b, text) in enumerate(chunks, 1):
        if mode == "watch":
            fn = lambda i=i, a=a, b=b: write_notes_visual(
                client, model, vid, title, i, total, a, b, language, delay, low_res)
        else:
            fn = lambda i=i, a=a, b=b, text=text: write_notes(
                client, model, title, i, total, a, b, text, language, delay, template)
        key = f"{vid}-{mode}-{language}-{model}-{int(a)}-{int(b)}"
        jobs.append(lambda key=key, fn=fn: cached_call(cache_dir, key, fn))
    return jobs


def run_parallel(jobs, workers, on_done=None):
    """Run zero-argument callables a few at a time.

    Returns (results, error): results keeps the original order (None where a job did not
    finish); error is the first exception raised, or None.
    """
    results = [None] * len(jobs)
    error = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(job): i for i, job in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except BaseException as e:
                if error is None:
                    error = e
                    for other in futs:
                        other.cancel()  # stop parts that have not started yet
                continue
            done += 1
            if on_done:
                on_done(done, len(jobs), i)
    return results, error


def assemble(title, video_id, parts):
    return (f"# {title}\n\nSource: https://www.youtube.com/watch?v={video_id}\n\n---\n\n"
            + "\n\n---\n\n".join(parts) + "\n")


# ----------------------------------------------------------------------------- command line


def main():
    p = argparse.ArgumentParser(description="YouTube playlist -> detailed lecture notes (Gemini)")
    p.add_argument("url", help="YouTube playlist or video link")
    p.add_argument("--mode", choices=["diagrams", "text", "watch"], default="diagrams",
                   help="diagrams = fast + flowcharts (default), text = fast, watch = Gemini watches the video")
    p.add_argument("--visual", action="store_true", help="same as --mode watch")
    p.add_argument("--out", default="notes", help="output folder (default: notes)")
    p.add_argument("--language", default="English", help="language of the notes (default: English)")
    p.add_argument("--chunk-minutes", type=float, default=None,
                   help="part length in minutes (default: 15, or 5 in watch mode)")
    p.add_argument("--low-res", action="store_true",
                   help="watch mode: faster and cheaper, but small text may be missed")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--delay", type=float, default=2, help="seconds to pause after each request (default: 2)")
    p.add_argument("--workers", type=int, default=3,
                   help="parts written at the same time (default: 3; use 1 if you hit rate limits)")
    args = p.parse_args()

    mode = "watch" if args.visual else args.mode
    chunk_minutes = args.chunk_minutes or default_chunk_minutes(mode)

    key = load_api_key()
    if not key:
        sys.exit("No Gemini API key found. Do ONE of these:\n"
                 "  1. PowerShell:  $env:GEMINI_API_KEY=\"your-key\"   (this window only)\n"
                 "  2. Put   GEMINI_API_KEY = \"your-key\"   in .streamlit/secrets.toml\n"
                 "  3. Save the key alone in a file named gemini_key.txt next to this script\n"
                 "Free key: https://aistudio.google.com/apikey")

    client = genai.Client(api_key=key)
    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, ".cache")

    videos = get_videos(args.url)
    print(f"Found {len(videos)} video(s). Mode: {MODE_LABELS[mode]}.")

    for n, (vid, title) in enumerate(videos, 1):
        path = os.path.join(args.out, f"{n:02d} - {safe_name(title)}.md")
        if os.path.exists(path):
            print(f"[{n}/{len(videos)}] Skipping (already done): {title}")
            continue
        print(f"[{n}/{len(videos)}] {title}")
        try:
            chunks = plan_chunks(vid, mode, chunk_minutes)
        except Exception as e:
            print(f"    Could not read this video ({type(e).__name__}); skipped.")
            continue

        jobs = build_jobs(client, args.model, vid, title, chunks, mode, args.language,
                          args.delay, args.low_res, cache_dir)

        def on_done(done, count, idx, chunks=chunks):
            a, b, _ = chunks[idx]
            print(f"    part {idx + 1}/{count} ({fmt_time(a)}-{fmt_time(b)}) done  [{done}/{count}]")

        print(f"    writing {len(chunks)} parts, {args.workers} at a time...")
        parts, err = run_parallel(jobs, args.workers, on_done)
        if isinstance(err, SystemExit):
            raise err
        if err is not None:
            print(f"\nStopped: {friendly_error(err)}")
            print("Run the same command again to continue; finished parts are not redone.")
            sys.exit(1)

        with open(path, "w", encoding="utf-8") as f:
            f.write(assemble(title, vid, parts))
        print(f"    saved -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
