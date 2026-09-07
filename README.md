# NiftyAI Mobile Cloud

Run file: `app.py`

Required cloud secrets / environment variables:
- `NIFTY_API_KEY`
- `NIFTY_ACCESS_TOKEN`

Do not commit API keys or access tokens into GitHub.

## Streamlit Community Cloud
1. Put these files in a GitHub repository.
2. Open https://share.streamlit.io from your phone and sign in with GitHub.
3. Create app, select the repo and set main file to `app.py`.
4. In app Settings > Secrets add your Zerodha credentials as environment-compatible secrets.
5. Deploy and open the generated URL on your phone.

Note: Zerodha access tokens expire and normally need renewal. Update the cloud secret when the token changes.
