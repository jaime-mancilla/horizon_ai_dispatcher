# Local testing with ngrok (for Twilio)

1. Run the server:
   ```bash
   pip install -r requirements.trackB.txt
   bash scripts/dev_server.sh
   ```
2. Start ngrok:
   ```bash
   ngrok http 8000
   ```
   Copy the `https://xxxxx.ngrok.io` URL.
3. In Twilio Console → Phone Numbers → Voice webhook (POST): `https://xxxxx.ngrok.io/twilio/voice`
4. Call your number and watch logs—the `/twilio/media` WebSocket will receive frames via ngrok (`wss` works through `https` tunnels).
