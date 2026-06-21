<div align="center">

# 📚 Book Recommendation System

A Flask-based book recommendation web app powered by a trained collaborative-filtering model, with Supabase-backed admin and contact features.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://book-recommend-system-1.onrender.com/)
[![Python](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-backend-black)](https://flask.palletsprojects.com/)
[![Supabase](https://img.shields.io/badge/database-Supabase-3FCF8E)](https://supabase.com/)

</div>

---

## 📖 Overview

This project serves book recommendations using model artifacts stored in `Model/`, with a Supabase backend that powers the admin dashboard and the public contact form.

**🔗 Live Website:** [book-recommend-system-1.onrender.com](https://book-recommend-system-1.onrender.com/)

---

## 📑 Table of Contents

- [Local Setup](#-local-setup)
- [Required Environment Variables](#-required-environment-variables)
- [Render Deployment — Adding the Supabase Admin Key](#-render-deployment--adding-the-supabase-admin-key)
- [Retraining Settings](#-retraining-settings)
- [Security Notes](#-security-notes)

---

## 🛠 Local Setup

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Copy `.env.example` to `.env` and fill in your own values.
4. Run the app:
   ```bash
   python app.py
   ```

---

## 🔑 Required Environment Variables

| Variable | Purpose | Notes |
|---|---|---|
| `FLASK_SECRET_KEY` | Flask session security | Required |
| `ADMIN_USERNAME` | Admin login username | Required |
| `ADMIN_PASSWORD` **or** `ADMIN_PASSWORD_HASH` | Admin login credential | One of the two is required |
| `SUPABASE_URL` **or** `NEXT_PUBLIC_SUPABASE_URL` | Full Supabase project URL | Alternative: use `Project_ID` below |
| `Project_ID` | Supabase project ref | Used to build `https://<Project_ID>.supabase.co` |
| `publishable_key`, `SUPABASE_ANON_KEY`, **or** `SUPABASE_PUBLISHABLE_KEY` | Public-safe Supabase key | Safe for Contact Us form & public reads |
| `SUPABASE_SERVICE_ROLE_KEY` **or** `SUPABASE_SECRET_KEY` | Backend-only admin key | ⚠️ Never expose to the client |

> **⚠️ Security Warning**
> `SUPABASE_SERVICE_ROLE_KEY` and `SUPABASE_SECRET_KEY` are **backend-only secrets**. Never place either key in frontend JavaScript, templates, or any client-side bundle.

> **ℹ️ Note on `publishable_key`**
> This key is safe for public operations such as the Contact Us form and public book reads. It **cannot** replace the service-role or secret key for admin add/edit/delete/import/retrain actions.

---

## 🚀 Render Deployment — Adding the Supabase Admin Key

If admin login succeeds but the dashboard displays:

```text
Missing Supabase admin configuration
```

or prompts for `SUPABASE_SERVICE_ROLE_KEY`, configure the backend service environment as follows:

1. Open the **Render Dashboard**.
2. Select the deployed **Book Recommendation** service.
3. Go to **Environment**.
4. Add `Project_ID` with your Supabase project ref — *or* add `SUPABASE_URL` with your full Supabase URL.
5. Add `publishable_key` with your Supabase publishable key.
6. Add `SUPABASE_SERVICE_ROLE_KEY`.
7. Paste your Supabase `service_role` key value.
8. **Save** and **redeploy**.

> **❌ Avoid:** Leaving `NEXT_PUBLIC_SUPABASE_URL` set to a placeholder such as `https://your-project-ref.supabase.co`.

If the URL is missing or still a placeholder, the backend will attempt to recover the project ref from a legacy Supabase JWT key — but the recommended production setup is to set `SUPABASE_URL` to the real project URL.

For newer Supabase projects, `SUPABASE_SECRET_KEY` may be used instead. Keep `SUPABASE_SERVICE_ROLE_KEY` functional if you're still using the legacy service-role key.

| If you only set... | What works | What doesn't |
|---|---|---|
| `Project_ID` + `publishable_key` | Contact Us, public book reads | Admin write/retrain buttons (need `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_SECRET_KEY`) |

---

## ⚙️ Retraining Settings

The retrain action pulls from the Supabase `books` and `ratings` tables and applies the same scale filters as the original notebook, keeping the similarity matrix small enough for a single web request.

| Setting | Default | Description |
|---|---|---|
| `POPULAR_MIN_RATINGS` | `250` | Minimum ratings for a book to be considered "popular" |
| `ACTIVE_USER_MIN_RATINGS` | `200` | Minimum ratings for an "active" user |
| `ACTIVE_BOOK_MIN_RATINGS` | `50` | Minimum ratings for an "active" book |

> **⚠️ Caution**
> You can lower these values in the server environment for a small test dataset, but **avoid setting them to `0` or `1`** with the full Book-Crossing dataset — this can produce an excessively large similarity matrix.

---

## 🔒 Security Notes

- ✅ Admin Supabase operations are performed **only** in `app.py` on the Flask server.
- ✅ The service-role/secret key is **never** sent to browser code.
- ✅ Browser-facing code only calls Flask routes — it never creates a Supabase admin client directly.

---

<div align="center">

Made with ❤️ for book lovers

</div>
