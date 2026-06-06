# Book Recommendation System

Flask app for book recommendations using the trained pickle files in `Model/` and Supabase-backed admin/contact features.

## Local Setup

1. Create a virtual environment.
2. Install dependencies from `requirement.txt`.
3. Copy `.env.example` to `.env` and fill in your own values.
4. Run:

```bash
python app.py
```

## Required Environment Variables

- `FLASK_SECRET_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD` or `ADMIN_PASSWORD_HASH`
- `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL`
- or `Project_ID`, which is used to build `https://<Project_ID>.supabase.co`
- `publishable_key`, `SUPABASE_ANON_KEY`, or `SUPABASE_PUBLISHABLE_KEY`
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_SECRET_KEY`

`SUPABASE_SERVICE_ROLE_KEY` and `SUPABASE_SECRET_KEY` are backend-only secrets. Do not put either key in frontend JavaScript, templates, or any client-side bundle.

`publishable_key` is safe for public operations such as the Contact Us form and public book reads. It cannot replace the service-role or secret key for admin add/edit/delete/import/retrain actions.

## Render Deployment: Add Supabase Admin Key

If admin login works but the dashboard shows:

```text
Missing Supabase admin configuration
```

or asks for `SUPABASE_SERVICE_ROLE_KEY`, configure the backend service environment:

1. Open the Render Dashboard.
2. Select the deployed Book Recommendation service.
3. Go to **Environment**.
4. Add key `Project_ID` with your Supabase project ref, or add `SUPABASE_URL` with your full Supabase URL.
5. Add key `publishable_key` with your Supabase publishable key.
6. Add key `SUPABASE_SERVICE_ROLE_KEY`.
7. Paste the Supabase `service_role` key value.
8. Save and redeploy.

Do not leave `NEXT_PUBLIC_SUPABASE_URL` set to a placeholder such as `https://your-project-ref.supabase.co`.
If the URL is missing or still a placeholder, the backend will try to recover the project ref from a legacy Supabase JWT key, but the best production setup is still to set `SUPABASE_URL` to the real project URL.

For newer Supabase projects, you may set `SUPABASE_SECRET_KEY` instead. Keep `SUPABASE_SERVICE_ROLE_KEY` working if you already use the legacy service-role key.

If you only set `Project_ID` and `publishable_key`, Contact Us and public book reads can work, but admin write/retrain buttons will still need `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_SECRET_KEY`.

## Retraining Settings

The retrain action uses the Supabase `books` and `ratings` tables and applies the same scale filters as the original notebook so the similarity matrix stays small enough for a web request:

- `POPULAR_MIN_RATINGS`, default `250`
- `ACTIVE_USER_MIN_RATINGS`, default `200`
- `ACTIVE_BOOK_MIN_RATINGS`, default `50`

You can lower these values in the server environment for a tiny test dataset, but avoid setting them to `0` or `1` with the full Book-Crossing dataset because that can produce an enormous similarity matrix.

## Security Notes

- Admin Supabase operations are performed only in `app.py` on the Flask server.
- The service-role or secret key is never sent to browser code.
- Browser-facing code only calls Flask routes; it does not create a Supabase admin client.
