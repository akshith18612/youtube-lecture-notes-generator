# YouTube Lecture Notes Generator — free deployment

Streamlit + Gemini + Supabase PostgreSQL. This version keeps the original note-generation modes and adds database persistence for generated notes and history.

## Public-demo privacy model

- Every visitor enters their **own Gemini API key**. The app does not write that key to a file or to Supabase.
- Supabase credentials live only in Streamlit server secrets.
- Generated notes are stored in `generation_history` with an anonymous UUID generated for the active Streamlit session. History is filtered by that UUID, so the UI does not show other active sessions' rows.
- **No login is implemented.** This means history is scoped to the active Streamlit session and may not be recoverable after the browser session ends. Database rows remain stored until deleted. If users need private history across visits/devices, add Supabase Auth and bind rows to authenticated user IDs before treating this as a production app.
- This is a public demo. Add abuse/rate controls before sharing widely. Users can submit large playlists and consume server resources even though they supply their own Gemini key.

## Files

- `app.py` — Streamlit UI and generation flow
- `playlist_notes_gemini.py` — original Gemini / YouTube helper logic
- `db.py` — Supabase read/write functions
- `supabase_schema.sql` — database table and index
- `.streamlit/config.toml` — visual configuration
- `requirements.txt` — Python dependencies

## 1. Create Supabase database

1. Create a project at <https://supabase.com/>.
2. Open **SQL Editor**, paste `supabase_schema.sql`, and run it.
3. In Project Settings / API (or API Keys, depending on dashboard layout), copy the project URL and the **service_role** secret key. Do not use the `service_role` key in browser code, and never commit it to GitHub. This app uses it only in Python on the Streamlit server.
4. Ensure the `generation_history` table is available through the Supabase Data API. Supabase's dashboard/settings can label this as Data API / exposed schemas.

## 2. Push to GitHub

Create a new repository and upload the files in this folder. Keep `.streamlit/secrets.toml`, `gemini_key.txt`, and `.env` out of Git; `.gitignore` already excludes them. Do not commit any real API keys.

## 3. Deploy on Streamlit Community Cloud

1. Visit <https://share.streamlit.io/> and sign in with GitHub.
2. Select **Create app**, your repository, branch `main`, and entrypoint `app.py`.
3. Open **Advanced settings → Secrets** and paste the following, replacing the placeholders:

```toml
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_SECRET"
```

4. Deploy. The app should open at a `*.streamlit.app` URL.

## 4. Local run (optional)

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` locally with the same two Supabase values, then run:

```bash
streamlit run app.py
```

Each visitor also needs a Gemini API key from <https://aistudio.google.com/apikey> entered in the app.

## 5. Smoke test

1. Open the app and enter your own Gemini key.
2. Generate notes from one public YouTube video with captions (start with `diagrams` or `text` mode).
3. Confirm the notes render and download.
4. Confirm a row appears in Supabase `generation_history`.
5. Confirm the row is visible in **Saved history** in the same active app session.
6. Check app logs if generation fails; YouTube transcript access can vary by video and hosting IP.

## Notes / limitations

- Free service tiers and API quotas can change. Gemini's free tier, Streamlit hosting, and Supabase database each have separate limits.
- `watch` mode can consume substantially more Gemini quota than transcript modes.
- YouTube may block transcript/video metadata requests from cloud IP addresses. The app retains the optional proxy field from the original project, but only use a proxy you are authorized to use.
- The original helper's local `.cache` is ephemeral on hosted deployments. This integration persists completed/partial notes, but it does not move the per-chunk retry cache into Supabase. A restart during generation may require redoing chunks that were not saved as a finished note.
- The database table has RLS enabled and intentionally has no anon/authenticated policies. Access is server-side using the service-role secret. Keep that secret private and do not expose it in any frontend.
