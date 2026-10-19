"""
Lecture notes from YouTube - web app (Streamlit + Gemini).

Run:   python -m streamlit run app.py
Needs playlist_notes_gemini.py in the same folder.
"""

import html
import io
import os
import re
import zipfile
import uuid

import streamlit as st
import streamlit.components.v1 as components
from google import genai

from playlist_notes_gemini import (
    DEFAULT_MODEL,
    assemble,
    build_jobs,
    default_chunk_minutes,
    fmt_time,
    friendly_error,
    get_videos,
    plan_chunks,
    run_parallel,
    safe_name,
)
from db import get_db, load_session_history, save_generation

st.set_page_config(page_title="Lecture Notes", page_icon="📝", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.hero h1 {font-size: 2.3rem; line-height: 1.15; margin: 0 0 .35rem 0; letter-spacing: -0.02em;}
.hero p {font-size: 1.05rem; color: #5b616b; margin: 0 0 1rem 0;}
.steps {display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 1.2rem;}
.step {background: #fff; border: 1px solid #e4e2d9; border-radius: 999px;
       padding: .3rem .85rem; font-size: .88rem; color: #3b4048;}
.step b {color: #3B5BDB; margin-right: .35rem;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e4e2d9;
       border-radius: 12px; padding: .7rem 1rem;}
.stButton > button[kind="primary"], .stDownloadButton > button {border-radius: 10px; font-weight: 600;}
.hint {color: #6b7280; font-size: .9rem; margin-top: .4rem;}
</style>
""",
    unsafe_allow_html=True,
)

LANGUAGES = ["English", "Telugu", "Hindi", "Tamil", "Kannada", "Malayalam", "Other..."]
MODE_OPTIONS = {
    "diagrams": (
        "⚡ Fast notes + diagrams (recommended)",
        "Reads the transcript and also draws flowcharts for the processes and structures the "
        "teacher explains. Quick and light on your free quota.",
    ),
    "text": (
        "📝 Fast, text only",
        "The same detailed notes, without diagrams.",
    ),
    "watch": (
        "🎥 Exact: watch the video",
        "Gemini watches the video and describes what is really drawn on screen. Slow and uses "
        "a lot of free quota. Public videos only.",
    ),
}
OUT_DIR = "notes"
MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.S)


# ----------------------------------------------------------------------------- helpers


def is_local():
    """True only when the app is opened on this same computer."""
    try:
        host = st.context.headers.get("Host", "").rsplit(":", 1)[0]
        return host in ("localhost", "127.0.0.1", "[::1]")
    except Exception:
        return False


def mermaid_html(code):
    return f"""
    <div class="mermaid" style="font-family:sans-serif">{html.escape(code)}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{startOnLoad: true, theme: 'neutral'}});
    </script>
    """


def render_notes(md):
    """Show markdown, drawing ```mermaid blocks as real diagrams."""
    pos = 0
    for m in MERMAID_RE.finditer(md):
        before = md[pos:m.start()]
        if before.strip():
            st.markdown(before)
        code = m.group(1).strip()
        components.html(mermaid_html(code), height=380, scrolling=True)
        with st.expander("Diagram source"):
            st.code(code, language="text")
        pos = m.end()
    rest = md[pos:]
    if rest.strip():
        st.markdown(rest)


if "results" not in st.session_state:
    st.session_state.results = {}  # filename -> markdown
if "demo_session_id" not in st.session_state:
    # Anonymous, per-live-session scope. This is not an account/login identity.
    st.session_state.demo_session_id = str(uuid.uuid4())

# ----------------------------------------------------------------------------- header

st.markdown(
    """
    <div class="hero">
      <h1>📝 Lecture notes from YouTube</h1>
      <p>Paste a playlist or video link. Get detailed notes that cover every point the teacher explained.</p>
    </div>
    <div class="steps">
      <span class="step"><b>1</b>Add your key</span>
      <span class="step"><b>2</b>Paste a link</span>
      <span class="step"><b>3</b>Download notes</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- API key

st.caption("Public demo: enter your own Gemini API key. It is used for this session only and is not saved to the database.")
with st.expander("🔑 Gemini API key", expanded=True):
    typed_key = st.text_input(
        "Gemini API key", type="password", placeholder="Paste your own Gemini API key",
        label_visibility="collapsed",
    )
    st.markdown("[Get a free key →](https://aistudio.google.com/apikey)")
api_key = (typed_key or "").strip()

# ----------------------------------------------------------------------------- advanced

with st.sidebar:
    st.subheader("Advanced")
    model = st.text_input("Gemini model", value=DEFAULT_MODEL,
                          help="If you get a model-not-found error, paste another model name from "
                               "aistudio.google.com/rate-limit.")
    workers = st.number_input("Parts written at once", 1, 6, 3,
                              help="Higher is faster. Lower it to 1 if you see rate-limit errors.")
    delay = st.number_input("Pause after each request (s)", 0.0, 30.0, 2.0)
    proxy_url = st.text_input("Proxy URL (optional)", type="password",
                              help="Only if YouTube blocks the server. Example: http://user:pass@host:port")
    st.caption("Finished parts are saved, so if you hit the free limit just click Generate again "
               "later and it continues where it stopped.")

# ----------------------------------------------------------------------------- inputs

url = st.text_input("YouTube link", placeholder="https://www.youtube.com/playlist?list=...",
                    label_visibility="collapsed")

mode = st.radio(
    "Mode", list(MODE_OPTIONS), format_func=lambda k: MODE_OPTIONS[k][0],
    captions=[v[1] for v in MODE_OPTIONS.values()], label_visibility="collapsed",
)
if mode == "diagrams":
    st.caption("Diagrams here are drawn from what the teacher says. For drawings that are only "
               "shown on screen and never explained aloud, use Exact mode.")

c1, c2, c3 = st.columns(3)
with c1:
    lang_choice = st.selectbox("Notes language", LANGUAGES)
    if lang_choice == "Other...":
        language = st.text_input("Language", placeholder="e.g. Marathi",
                                 label_visibility="collapsed") or "English"
    else:
        language = lang_choice
with c2:
    max_videos = st.number_input("Max videos", 1, 50, 3, help="Keep small on the free tier.")
with c3:
    chunk_minutes = st.slider(
        "Part length (min)", 3, 30, default_chunk_minutes(mode), key=f"chunk_{mode}",
        help="Shorter parts give more detailed notes but use more requests.")

low_res = False
if mode == "watch":
    low_res = st.checkbox("Faster, lower-quality video (may miss small text)", value=False)

go = st.button("Generate notes", type="primary", disabled=not (url and api_key),
               use_container_width=True)
if not api_key:
    st.markdown('<div class="hint">Enter your own Gemini API key above to start.</div>',
                unsafe_allow_html=True)
elif not url:
    st.markdown('<div class="hint">Paste a YouTube link above.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- generation

if go:
    st.session_state.results = {}
    client = genai.Client(api_key=api_key)
    cache_dir = os.path.join(OUT_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with st.spinner("Reading the link..."):
            videos = get_videos(url)
    except Exception as e:
        st.error(f"Could not read that link ({type(e).__name__}). Check that it is a public "
                 "YouTube video or playlist.")
        st.stop()

    videos = videos[: int(max_videos)]
    overall_text = st.empty()
    overall = st.progress(0.0)

    for n, (vid, title) in enumerate(videos, 1):
        overall_text.markdown(f"**Video {n} of {len(videos)}**")
        with st.status(title, expanded=True) as status:
            try:
                chunks = plan_chunks(vid, mode, chunk_minutes, proxy_url or None)
            except Exception as e:
                status.update(label=f"{title} - could not read", state="error")
                if mode == "watch":
                    st.warning(f"Could not read this video's length ({type(e).__name__}).")
                else:
                    st.warning(
                        f"Could not get the transcript ({type(e).__name__}). The video may have no "
                        "captions, or YouTube is blocking this server. Try Exact mode, which does "
                        "not need transcripts, or add a proxy in Advanced.")
                overall.progress(n / len(videos))
                continue

            total = len(chunks)
            jobs = build_jobs(client, model, vid, title, chunks, mode, language,
                              float(delay), low_res, cache_dir)
            part_bar = st.progress(0.0)
            status.write(f"Writing {total} parts, {int(workers)} at a time...")

            def on_done(done, count, idx, _bar=part_bar, _status=status, _chunks=chunks):
                a, b, _ = _chunks[idx]
                _bar.progress(done / count)
                _status.write(f"Part {idx + 1} of {count} ({fmt_time(a)} to {fmt_time(b)}) done")

            parts, err = run_parallel(jobs, int(workers), on_done)
            done_parts = [part for part in parts if part]

            if err is not None:
                status.update(label=f"{title} - stopped", state="error")
                st.error(friendly_error(err))
                st.info("Finished parts are cached locally for this running instance. Click Generate notes again later to retry.")
                if done_parts:
                    partial_md = assemble(title, vid, done_parts)
                    partial_name = f"{n:02d} - {safe_name(title)} (partial).md"
                    st.session_state.results[partial_name] = partial_md
                    try:
                        save_generation(
                            session_id=st.session_state.demo_session_id,
                            video_id=vid, title=title,
                            source_url=f"https://www.youtube.com/watch?v={vid}",
                            mode=mode, language=language, notes_markdown=partial_md, status="partial",
                        )
                    except Exception as db_error:
                        st.warning(f"Partial notes are available for download, but could not be saved to the database ({type(db_error).__name__}).")
                overall.progress(n / len(videos))
                break

            md = assemble(title, vid, done_parts)
            name = f"{n:02d} - {safe_name(title)}.md"
            with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
                f.write(md)
            st.session_state.results[name] = md
            try:
                save_generation(
                    session_id=st.session_state.demo_session_id,
                    video_id=vid, title=title,
                    source_url=f"https://www.youtube.com/watch?v={vid}",
                    mode=mode, language=language, notes_markdown=md, status="completed",
                )
            except Exception as db_error:
                st.warning(f"Notes were generated, but could not be saved to the database ({type(db_error).__name__}). Check Supabase setup.")
            status.update(label=f"{title} - done", state="complete", expanded=False)
        overall.progress(n / len(videos))
    overall_text.empty()

# ----------------------------------------------------------------------------- results

results = st.session_state.results
if results:
    st.divider()
    st.subheader("Your notes")

    total_words = sum(len(m.split()) for m in results.values())
    total_diagrams = sum(m.count("```mermaid") for m in results.values())
    cols = st.columns(3)
    cols[0].metric("Videos", len(results))
    cols[1].metric("Words", f"{total_words:,}")
    cols[2].metric("Diagrams", total_diagrams)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, md in results.items():
            z.writestr(name, md)
    st.download_button("⬇️ Download all (zip)", buf.getvalue(), "lecture_notes.zip",
                       "application/zip", use_container_width=True)
    st.caption("Your notes are saved in this active session and, when Supabase is configured, in the database.")

    names = list(results.keys())
    chosen = names[0] if len(names) == 1 else st.selectbox("Open notes", names)
    md = results[chosen]
    st.download_button("Download this file (.md)", md, chosen, "text/markdown", key=f"dl-{chosen}")
    with st.container(border=True):
        render_notes(md)


# ----------------------------------------------------------------------------- persistent database history

st.divider()
st.subheader("Saved history")
if get_db() is None:
    st.info("Database history is not connected yet. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to Streamlit secrets after creating the Supabase table.")
else:
    try:
        history = load_session_history(st.session_state.demo_session_id)
        if not history:
            st.caption("No saved generations for this active session yet.")
        else:
            history_labels = [
                f"{row['created_at'][:19].replace('T', ' ')} · {row['title']} ({row['status']})"
                for row in history
            ]
            chosen_history = st.selectbox("Open a previously saved note", range(len(history)), format_func=lambda i: history_labels[i])
            selected = history[chosen_history]
            st.caption(f"Video: {selected['source_url']} · Mode: {selected['mode']} · Language: {selected['language']}")
            st.download_button(
                "Download saved note (.md)", selected["notes_markdown"],
                file_name=f"{safe_name(selected['title'])}.md", mime="text/markdown",
                key=f"history-download-{selected['id']}",
            )
            with st.container(border=True):
                render_notes(selected["notes_markdown"])
    except Exception as db_error:
        st.error(f"Could not load saved history ({type(db_error).__name__}). Check the Supabase URL, key, table, and Data API settings.")
