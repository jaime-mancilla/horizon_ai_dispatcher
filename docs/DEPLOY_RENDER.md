# Deploy to Render — Track B (Twilio Media Streams)

## Option 1: Native (recommended to start)
1. Commit `render.yaml` and `requirements.trackB.txt`.
2. Push to GitHub.
3. In Render → "New" → "Web Service" → select your repo.
4. Render will read `render.yaml` and use:
   - Build: `pip install -r requirements.trackB.txt`
   - Start: `uvicorn app.server:app --host 0.0.0.0 --port $PORT`
5. Set environment variables in Render **Environment** tab (no secrets in repo).
6. Deploy, then visit `https://<your-render-app>.onrender.com/healthz`.

## Option 2: Docker (for full control)
1. Keep `Dockerfile` and choose "Deploy an existing Dockerfile" in Render.
2. No build/start commands needed—Render builds from `Dockerfile`.
3. Same env vars apply.

> Note: Free plans may spin down. For inbound calls, use a plan that stays warm or call once to "wake" the dyno before testing.
