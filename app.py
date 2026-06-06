import base64
import binascii
import json
import logging
import os
import pickle
import re
from functools import wraps
from urllib import error, parse, request

import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request as flask_request,
    session,
    url_for,
)
from rapidfuzz import fuzz, process
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
SUPABASE_PLACEHOLDER_TOKENS = {
    "your-project-ref",
    "replace-with",
    "your-supabase-url",
}
DEFAULT_POPULAR_MIN_RATINGS = 250
DEFAULT_ACTIVE_USER_MIN_RATINGS = 200
DEFAULT_ACTIVE_BOOK_MIN_RATINGS = 50


def load_dotenv_file():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def env_value(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    lowercase_names = {name.lower() for name in names}
    for key, value in os.environ.items():
        if key.lower() in lowercase_names and value:
            return value

    return ""


def env_int(name, default):
    value = env_value(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        app.logger.warning("Invalid integer for %s=%r; using %s.", name, value, default)
        return default


def is_placeholder_value(value):
    normalized = str(value or "").strip().lower()
    return any(token in normalized for token in SUPABASE_PLACEHOLDER_TOKENS)


def normalize_supabase_url(value, source_name):
    raw_url = str(value or "").strip()
    if not raw_url or is_placeholder_value(raw_url):
        raise RuntimeError(
            f"Invalid Supabase URL in {source_name}. Replace the placeholder with your real project URL, "
            "for example https://abc123def456ghi789jk.supabase.co."
        )

    if "://" not in raw_url:
        raw_url = f"https://{raw_url}"

    parsed = parse.urlparse(raw_url)
    hostname = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise RuntimeError(
            f"Invalid Supabase URL in {source_name}. Use the full project URL, "
            "for example https://abc123def456ghi789jk.supabase.co."
        )

    if is_placeholder_value(hostname):
        raise RuntimeError(
            f"Invalid Supabase URL in {source_name}. Replace the placeholder host with your real Supabase project ref."
        )

    path = parsed.path.rstrip("/")
    if path and path != "/":
        raise RuntimeError(
            f"Invalid Supabase URL in {source_name}. Do not include a path; use only the project origin."
        )

    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def normalize_supabase_project_url(value, source_name):
    project_id = str(value or "").strip()
    if not project_id or is_placeholder_value(project_id):
        raise RuntimeError(
            f"Invalid Supabase project ref in {source_name}. Replace it with your real project ref."
        )

    if "://" in project_id:
        return normalize_supabase_url(project_id, source_name)

    project_id = project_id.rstrip("/")
    if "/" in project_id:
        raise RuntimeError(
            f"Invalid Supabase project ref in {source_name}. Use only the project ref, not a URL path."
        )

    return normalize_supabase_url(f"https://{project_id}.supabase.co", source_name)


def supabase_project_ref_from_key(key):
    parts = str(key or "").split(".")
    if len(parts) < 2:
        return ""

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)

    try:
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(decoded.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return ""

    project_ref = str(data.get("ref") or "").strip()
    return project_ref if project_ref and not is_placeholder_value(project_ref) else ""


def supabase_url_from_key(key):
    project_ref = supabase_project_ref_from_key(key)
    if not project_ref:
        return ""

    return normalize_supabase_project_url(project_ref, "Supabase API key project ref")


load_dotenv_file()

app = Flask(__name__)
app.secret_key = env_value("FLASK_SECRET_KEY") or "book-recommendation-dev-key"
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as file:
        return pickle.load(file)


def clean_value(value, default="Not available"):
    if value is None:
        return default

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default

    return text


def normalize_title(value):
    text = clean_value(value, "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_model_alignment():
    expected_shape = (len(pt.index), len(pt.index))
    actual_shape = getattr(similarity_scores, "shape", None)

    if actual_shape != expected_shape:
        message = (
            "Model file mismatch: similarity_scores.pkl shape "
            f"{actual_shape} does not match pt.pkl title count {expected_shape}. "
            "Regenerate similarity_scores.pkl from the exact pt.pkl before serving recommendations."
        )
        app.logger.error(message)
        raise RuntimeError(message)

    app.logger.info(
        "Model alignment verified: pt titles=%s, similarity_scores shape=%s",
        len(pt.index),
        actual_shape,
    )


def refresh_model_cache():
    global popular_df, books_df, pt, similarity_scores, MODEL_TITLES, TITLE_LOOKUP, NORMALIZED_TITLE_LOOKUP

    popular_df = load_pickle("popular.pkl")
    books_df = load_pickle("books.pkl")
    pt = load_pickle("pt.pkl")
    similarity_scores = load_pickle("similarity_scores.pkl")

    validate_model_alignment()
    MODEL_TITLES = [str(title) for title in pt.index]
    TITLE_LOOKUP = {title.lower(): index for index, title in enumerate(MODEL_TITLES)}
    NORMALIZED_TITLE_LOOKUP = {normalize_title(title): title for title in MODEL_TITLES}


refresh_model_cache()


def image_for(row):
    for column in ("Image-URL-L", "Image-URL-M", "Image-URL-S", "image_url_l", "image_url_m", "image_url_s"):
        if column in row and clean_value(row[column], ""):
            return clean_value(row[column])

    return "https://placehold.co/320x480/111827/f8fafc?text=Book+Cover"


def find_book_row(title=None, author=None, isbn=None):
    if isbn:
        matches = books_df[books_df["ISBN"].astype(str) == str(isbn)]
        if not matches.empty:
            return matches.iloc[0]

    if title:
        title_matches = books_df[books_df["Book-Title"].astype(str).str.lower() == str(title).lower()]
        if author and not title_matches.empty:
            author_matches = title_matches[
                title_matches["Book-Author"].astype(str).str.lower() == str(author).lower()
            ]
            if not author_matches.empty:
                return author_matches.iloc[0]

        if not title_matches.empty:
            return title_matches.iloc[0]

    return None


def book_from_row(row, include_stats=False):
    book = {
        "isbn": clean_value(row.get("ISBN", row.get("isbn", ""))),
        "title": clean_value(row.get("Book-Title", row.get("book_title", ""))),
        "author": clean_value(row.get("Book-Author", row.get("book_author", ""))),
        "publisher": clean_value(row.get("Publisher", row.get("publisher", ""))),
        "year": clean_value(row.get("Year-Of-Publication", row.get("year_of_publication", ""))),
        "image": image_for(row),
    }

    if include_stats:
        book["num_ratings"] = int(float(row.get("num_ratings", 0)))
        book["avg_ratings"] = round(float(row.get("avg_ratings", 0)), 2)

    if not book["isbn"] or book["isbn"] == "Not available":
        full_row = find_book_row(book["title"], book["author"])
        if full_row is not None:
            book.update(book_from_row(full_row))
            if include_stats:
                book["num_ratings"] = int(float(row.get("num_ratings", 0)))
                book["avg_ratings"] = round(float(row.get("avg_ratings", 0)), 2)

    return book


def get_popular_books():
    return [book_from_row(row, include_stats=True) for _, row in popular_df.head(50).iterrows()]


def fuzzy_title_matches(query, limit=5, cutoff=55):
    if not query:
        return []

    scored_titles = []
    normalized_query = normalize_title(query)

    for title in MODEL_TITLES:
        normalized_title = normalize_title(title)
        score = max(
            fuzz.partial_ratio(normalized_query, normalized_title),
            fuzz.token_set_ratio(normalized_query, normalized_title),
            fuzz.WRatio(normalized_query, normalized_title) * 0.8,
        )
        if score >= cutoff:
            scored_titles.append({"title": title, "score": round(score, 1)})

    return sorted(scored_titles, key=lambda item: item["score"], reverse=True)[:limit]


def best_title_match(query):
    if not query:
        return None, 0

    normalized_query = normalize_title(query)
    if normalized_query in NORMALIZED_TITLE_LOOKUP:
        return NORMALIZED_TITLE_LOOKUP[normalized_query], 100

    matches = fuzzy_title_matches(query, limit=1, cutoff=0)
    if not matches:
        return None, 0

    return matches[0]["title"], matches[0]["score"]


def recommend_books(book_name, limit=8):
    query = clean_value(book_name, "")
    normalized = query.lower()
    debug_info = {
        "searched_book": query,
        "matched_book": "",
        "book_index": "",
        "top_scores": [],
    }

    if not normalized:
        return [], "Please enter a book name to get recommendations.", [], "", debug_info

    matched_title = query if normalized in TITLE_LOOKUP else None
    did_you_mean = ""

    if matched_title is None:
        fuzzy_title, score = best_title_match(query)
        if fuzzy_title and score >= 68:
            matched_title = fuzzy_title
            did_you_mean = fuzzy_title
        else:
            suggestions = [item["title"] for item in fuzzy_title_matches(query)]
            catalog_match = find_book_row(title=query) or get_supabase_book_by_title(query)
            if catalog_match is not None:
                return (
                    [],
                    "This book is in the catalog, but recommendations will be available after model retraining.",
                    suggestions,
                    "",
                    debug_info,
                )
            return [], "Book not found in the trained recommendation model.", suggestions, "", debug_info

    book_index = TITLE_LOOKUP.get(matched_title.lower())
    if book_index is None:
        return (
            [],
            "This book is not yet included in the active model. Recommendations will be available after model retraining.",
            [],
            did_you_mean,
            debug_info,
        )

    scores = sorted(
        enumerate(similarity_scores[book_index]),
        key=lambda item: item[1],
        reverse=True,
    )
    top_scores = [
        {
            "index": int(index),
            "title": MODEL_TITLES[index],
            "score": round(float(score), 4),
            "confidence": round(float(score) * 100, 2),
        }
        for index, score in scores[1 : limit + 1]
    ]
    debug_info = {
        "searched_book": query,
        "matched_book": matched_title,
        "book_index": int(book_index),
        "top_scores": top_scores,
    }

    app.logger.info(
        "Recommendation debug | searched=%r matched=%r index=%s top_scores=%s",
        query,
        matched_title,
        book_index,
        [(item["index"], item["title"], item["score"]) for item in top_scores],
    )

    recommendations = []
    seen_titles = set()

    for index, score in scores[1:]:
        title = MODEL_TITLES[index]
        if title.lower() in seen_titles:
            continue

        row = find_book_row(title=title)
        if row is None:
            continue

        book = book_from_row(row)
        book["source_title"] = title
        book["similarity_score"] = round(float(score), 4)
        book["confidence"] = round(float(score) * 100, 2)
        recommendations.append(book)
        seen_titles.add(title.lower())

        if len(recommendations) >= limit:
            break

    if not recommendations:
        return (
            [],
            "No similar books were returned by the trained model for this title.",
            [],
            did_you_mean,
            debug_info,
        )

    return recommendations, "", [], did_you_mean, debug_info


def selected_book_from_debug(debug_info):
    matched_title = debug_info.get("matched_book") if debug_info else ""

    if not matched_title:
        return None

    row = find_book_row(title=matched_title)
    if row is None:
        return None

    return book_from_row(row)


def supabase_url():
    url_sources = [
        ("SUPABASE_URL", env_value("SUPABASE_URL")),
        ("NEXT_PUBLIC_SUPABASE_URL", env_value("NEXT_PUBLIC_SUPABASE_URL")),
    ]
    project_sources = [
        ("Project_ID", env_value("Project_ID", "project_id")),
        ("PROJECT_ID", env_value("PROJECT_ID")),
        ("SUPABASE_PROJECT_ID", env_value("SUPABASE_PROJECT_ID")),
    ]
    has_configured_value = any(value for _, value in [*url_sources, *project_sources])

    if not has_configured_value:
        raise RuntimeError(
            "Missing Supabase URL. Set SUPABASE_URL, NEXT_PUBLIC_SUPABASE_URL, or Project_ID in the server environment."
        )

    errors = []
    for source_name, value in url_sources:
        if not value:
            continue
        try:
            return normalize_supabase_url(value, source_name)
        except RuntimeError as exc:
            errors.append(str(exc))

    for source_name, value in project_sources:
        if not value:
            continue
        try:
            return normalize_supabase_project_url(value, source_name)
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError(" ".join(errors))


def supabase_key(admin=False):
    if admin:
        return (
            env_value(
                "SUPABASE_SERVICE_ROLE_KEY",
                "SUPABASE_SECRET_KEY",
                "service_role_key",
                "secret_key",
            )
            or ""
        )

    return (
        env_value(
            "publishable_key",
            "PUBLISHABLE_KEY",
            "SUPABASE_ANON_KEY",
            "SUPABASE_PUBLISHABLE_KEY",
            "SUPABASE_KEY",
        )
        or ""
    )


def require_supabase_key(admin=False):
    key = supabase_key(admin=admin)
    if not key:
        if admin:
            raise RuntimeError(
                "Missing Supabase admin key. Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SECRET_KEY "
                "in the server environment. Do not expose this key in frontend code."
            )
        raise RuntimeError(
            "Missing Supabase public key. Set publishable_key, SUPABASE_ANON_KEY, SUPABASE_PUBLISHABLE_KEY, or SUPABASE_KEY."
        )
    return key


def require_supabase_config(admin=False):
    missing = []
    url_error = ""
    try:
        url = supabase_url()
    except RuntimeError as exc:
        url = ""
        url_error = str(exc)
    key = supabase_key(admin=admin)

    if not url and key:
        url = supabase_url_from_key(key)

    if not url and not missing:
        missing.append(url_error or "SUPABASE_URL, NEXT_PUBLIC_SUPABASE_URL, or Project_ID")

    if not key:
        if admin:
            missing.append("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SECRET_KEY")
        else:
            missing.append("publishable_key, SUPABASE_ANON_KEY, SUPABASE_PUBLISHABLE_KEY, or SUPABASE_KEY")

    if missing:
        if admin:
            raise RuntimeError(
                "Missing Supabase admin configuration: "
                + "; ".join(missing)
                + ". Add these variables in the server environment. Never expose service_role/secret keys in frontend code."
            )

        raise RuntimeError(
            "Missing Supabase configuration: "
            + "; ".join(missing)
            + ". Add these variables in the server environment."
        )

    return url.rstrip("/"), key


def supabase_request(path, method="GET", payload=None, admin=False, headers=None):
    # Service-role/secret keys are used only here in backend Python code.
    url, key = require_supabase_config(admin=admin)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req_headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    if headers:
        req_headers.update(headers)

    req = request.Request(
        f"{url}/rest/v1/{path}",
        data=data,
        method=method,
        headers=req_headers,
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="ignore")
            return json.loads(body) if body else None
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if '"57014"' in detail or "statement timeout" in detail.lower():
            raise RuntimeError("Supabase query timed out. Please reduce query size or add indexes.") from exc
        raise RuntimeError(f"Supabase request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Supabase: {exc.reason}") from exc


def supabase_bulk_upsert(table_name, rows, conflict_key=None, batch_size=500):
    if not rows:
        return 0

    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        path = table_name
        headers = {"Prefer": "return=minimal"}

        if conflict_key:
            path = f"{table_name}?on_conflict={conflict_key}"
            headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

        supabase_request(path, method="POST", payload=batch, admin=True, headers=headers)
        inserted += len(batch)

    return inserted


def fetch_supabase_table(table_name, order=None, limit=None, admin=True, select="*", page_size=1000, offset=0):
    rows = []
    current_offset = max(0, int(offset))
    page_size = max(1, min(int(page_size), 1000))
    remaining = int(limit) if limit else None

    while True:
        batch_limit = min(page_size, remaining) if remaining else page_size
        query = f"select={parse.quote(select, safe='*,')}"
        if order:
            query += f"&order={parse.quote(order, safe='.,')}"
        query += f"&limit={batch_limit}&offset={current_offset}"

        batch = supabase_request(f"{table_name}?{query}", admin=admin) or []
        rows.extend(batch)

        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                break

        if len(batch) < batch_limit:
            break

        current_offset += batch_limit

    return rows


def get_supabase_books(order="created_at.desc", limit=50, admin=False, select="*", offset=0):
    return fetch_supabase_table("books", order=order, limit=limit, admin=admin, select=select, offset=offset)


def get_supabase_table_count(table_name, admin=True):
    count_column = {
        "books": "isbn",
        "ratings": "id",
        "users": "user_id",
    }.get(table_name, "*")
    url, key = require_supabase_config(admin=admin)
    req = request.Request(
        f"{url}/rest/v1/{table_name}?select={parse.quote(count_column)}&limit=1",
        method="GET",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "count=exact",
        },
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if '"57014"' in detail or "statement timeout" in detail.lower():
            raise RuntimeError("Supabase query timed out. Please reduce query size or add indexes.") from exc
        raise RuntimeError(f"Supabase request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Supabase: {exc.reason}") from exc

    if "/" not in content_range:
        return 0

    total = content_range.rsplit("/", 1)[-1]
    return int(total) if total.isdigit() else 0


def search_supabase_books(query, limit=50, offset=0):
    selected = "isbn,book_title,book_author,year_of_publication,publisher"
    if not query:
        return get_supabase_books(order="book_title.asc", limit=limit, admin=True, select=selected, offset=offset)

    encoded = parse.quote(f"*{query}*", safe="*")
    path = (
        "books?"
        f"select={parse.quote(selected, safe=',')}"
        f"&or=(book_title.ilike.{encoded},book_author.ilike.{encoded},publisher.ilike.{encoded},isbn.ilike.{encoded})"
        "&order=book_title.asc"
        f"&limit={limit}"
        f"&offset={offset}"
    )
    return supabase_request(path, admin=True) or []


def upsert_supabase_book(form):
    isbn = clean_value(form.get("isbn", ""), "")
    if not isbn:
        raise RuntimeError("ISBN is required.")

    payload = {
        "isbn": isbn,
        "book_title": clean_value(form.get("book_title", ""), ""),
        "book_author": clean_value(form.get("book_author", ""), ""),
        "year_of_publication": clean_value(form.get("year_of_publication", ""), ""),
        "publisher": clean_value(form.get("publisher", ""), ""),
        "image_url_s": clean_value(form.get("image_url_s", ""), ""),
        "image_url_m": clean_value(form.get("image_url_m", ""), ""),
        "image_url_l": clean_value(form.get("image_url_l", ""), ""),
    }

    if not payload["book_title"] or not payload["book_author"]:
        raise RuntimeError("Book title and author are required.")

    supabase_request(
        "books?on_conflict=isbn",
        method="POST",
        payload=payload,
        admin=True,
        headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
    )


def delete_supabase_book(isbn):
    encoded = parse.quote(str(isbn), safe="")
    supabase_request(f"books?isbn=eq.{encoded}", method="DELETE", admin=True, headers={"Prefer": "return=minimal"})


def get_supabase_book(isbn):
    encoded = parse.quote(str(isbn), safe="")
    rows = supabase_request(f"books?select=*&isbn=eq.{encoded}&limit=1", admin=True) or []
    return rows[0] if rows else None


def get_supabase_book_by_title(title):
    encoded = parse.quote(str(title), safe="")

    try:
        rows = supabase_request(f"books?select=*&book_title=eq.{encoded}&limit=1", admin=False) or []
    except RuntimeError:
        try:
            rows = supabase_request(f"books?select=*&book_title=eq.{encoded}&limit=1", admin=True) or []
        except RuntimeError:
            rows = []

    return rows[0] if rows else None


def store_contact_submission(name, email, message):
    payload = {"name": name, "email": email, "message": message}
    supabase_request(
        "contact_submissions",
        method="POST",
        payload=payload,
        admin=False,
        headers={"Prefer": "return=minimal"},
    )


def admin_password_hash():
    configured = os.environ.get("ADMIN_PASSWORD_HASH")
    if configured:
        return configured

    password = os.environ.get("ADMIN_PASSWORD", "admin123")
    return generate_password_hash(password)


def admin_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please log in as an admin to continue.", "warning")
            return redirect(url_for("admin_login", next=flask_request.path))

        return view(*args, **kwargs)

    return wrapped


def train_models_from_supabase():
    progress = ["Fetching books from Supabase."]
    popular_min_ratings = env_int("POPULAR_MIN_RATINGS", DEFAULT_POPULAR_MIN_RATINGS)
    active_user_min_ratings = env_int("ACTIVE_USER_MIN_RATINGS", DEFAULT_ACTIVE_USER_MIN_RATINGS)
    active_book_min_ratings = env_int("ACTIVE_BOOK_MIN_RATINGS", DEFAULT_ACTIVE_BOOK_MIN_RATINGS)
    book_columns = ",".join(
        [
            "isbn",
            "book_title",
            "book_author",
            "year_of_publication",
            "publisher",
            "image_url_s",
            "image_url_m",
            "image_url_l",
        ]
    )
    rating_columns = "user_id,isbn,book_rating"

    supabase_books = fetch_supabase_table(
        "books",
        order="book_title.asc",
        admin=True,
        select=book_columns,
        page_size=1000,
    )
    if not supabase_books:
        raise RuntimeError("No books found in Supabase. Add or import books before retraining.")

    progress.append(f"Fetched {len(supabase_books)} books in batches of 1000.")
    supabase_ratings = fetch_supabase_table(
        "ratings",
        order="id.asc",
        admin=True,
        select=rating_columns,
        page_size=1000,
    )
    if not supabase_ratings:
        raise RuntimeError("No ratings found in Supabase. Collaborative retraining requires rating rows.")

    progress.append(f"Fetched {len(supabase_ratings)} ratings in batches of 1000.")

    new_books_df = pd.DataFrame(supabase_books).rename(
        columns={
            "isbn": "ISBN",
            "book_title": "Book-Title",
            "book_author": "Book-Author",
            "year_of_publication": "Year-Of-Publication",
            "publisher": "Publisher",
            "image_url_s": "Image-URL-S",
            "image_url_m": "Image-URL-M",
            "image_url_l": "Image-URL-L",
        }
    )
    new_books_df = new_books_df[
        ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"]
    ].drop_duplicates("ISBN")

    ratings_df = pd.DataFrame(supabase_ratings).rename(
        columns={"user_id": "User-ID", "isbn": "ISBN", "book_rating": "Book-Rating"}
    )
    ratings_df["Book-Rating"] = pd.to_numeric(ratings_df["Book-Rating"], errors="coerce").fillna(0)
    ratings_df["User-ID"] = pd.to_numeric(ratings_df["User-ID"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["User-ID"])
    ratings_df["User-ID"] = ratings_df["User-ID"].astype(int)

    merged = ratings_df.merge(new_books_df, on="ISBN")
    if merged.empty:
        raise RuntimeError("Ratings did not match any Supabase books. Model files were not changed.")

    progress.append("Building popularity data.")
    num_rating_df = merged.groupby("Book-Title").count()["Book-Rating"].reset_index()
    num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)
    avg_rating_df = merged.groupby("Book-Title").mean(numeric_only=True)["Book-Rating"].reset_index()
    avg_rating_df.rename(columns={"Book-Rating": "avg_ratings"}, inplace=True)
    new_popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
    new_popular_df = (
        new_popular_df[new_popular_df["num_ratings"] >= popular_min_ratings]
        .sort_values("avg_ratings", ascending=False)
        .head(50)
    )
    new_popular_df = new_popular_df.merge(new_books_df, on="Book-Title").drop_duplicates("Book-Title")
    new_popular_df = new_popular_df[
        ["Book-Title", "Book-Author", "Image-URL-M", "Year-Of-Publication", "num_ratings", "avg_ratings"]
    ]

    progress.append("Building user-book pivot table.")
    x = merged.groupby("User-ID").count()["Book-Rating"] > active_user_min_ratings
    active_users = x[x].index
    filtered = merged[merged["User-ID"].isin(active_users)]
    progress.append(f"Kept {len(active_users)} users with more than {active_user_min_ratings} ratings.")

    y = filtered.groupby("Book-Title").count()["Book-Rating"] >= active_book_min_ratings
    active_books = y[y].index
    final_ratings = filtered[filtered["Book-Title"].isin(active_books)]
    progress.append(f"Kept {len(active_books)} books with at least {active_book_min_ratings} ratings.")

    if final_ratings.empty:
        raise RuntimeError(
            "No ratings remained after retraining filters. Lower ACTIVE_USER_MIN_RATINGS or ACTIVE_BOOK_MIN_RATINGS."
        )

    new_pt = final_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating").fillna(0)
    if len(new_pt.index) < 2:
        raise RuntimeError("At least two rated books are required to build similarity scores.")

    progress.append(f"Calculating similarity scores for {len(new_pt.index)} books.")
    matrix = new_pt.to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = matrix / norms
    new_similarity_scores = np.dot(normalized, normalized.T)

    progress.append("Replacing model files.")
    model_outputs = {
        "books.pkl": new_books_df,
        "popular.pkl": new_popular_df,
        "pt.pkl": new_pt,
        "similarity_scores.pkl": new_similarity_scores,
    }

    for filename, obj in model_outputs.items():
        temp_path = os.path.join(MODEL_DIR, f"{filename}.tmp")
        final_path = os.path.join(MODEL_DIR, filename)
        with open(temp_path, "wb") as file:
            pickle.dump(obj, file)
        os.replace(temp_path, final_path)

    refresh_model_cache()
    progress.append("Model cache refreshed successfully.")
    return progress


def import_current_books_to_supabase():
    rows = []

    for _, row in books_df.iterrows():
        isbn = clean_value(row.get("ISBN", ""), "")
        title = clean_value(row.get("Book-Title", ""), "")
        author = clean_value(row.get("Book-Author", ""), "")

        if not isbn or not title or not author:
            continue

        rows.append(
            {
                "isbn": isbn,
                "book_title": title,
                "book_author": author,
                "year_of_publication": clean_value(row.get("Year-Of-Publication", ""), ""),
                "publisher": clean_value(row.get("Publisher", ""), ""),
                "image_url_s": clean_value(row.get("Image-URL-S", ""), ""),
                "image_url_m": clean_value(row.get("Image-URL-M", ""), ""),
                "image_url_l": clean_value(row.get("Image-URL-L", ""), ""),
            }
        )

    return supabase_bulk_upsert("books", rows, conflict_key="isbn")


def import_dataset_ratings_to_supabase():
    ratings_path = os.path.join(DATASET_DIR, "Ratings.csv")
    if not os.path.exists(ratings_path):
        raise RuntimeError("Dataset/Ratings.csv was not found.")

    imported = 0

    for ratings in pd.read_csv(ratings_path, chunksize=1000):
        rows = []
        for _, row in ratings.iterrows():
            user_id = row.get("User-ID")
            isbn = clean_value(row.get("ISBN", ""), "")

            if pd.isna(user_id) or not isbn:
                continue

            rows.append(
                {
                    "user_id": int(user_id),
                    "isbn": isbn,
                    "book_rating": float(row.get("Book-Rating", 0) or 0),
                }
            )

        imported += supabase_bulk_upsert("ratings", rows, conflict_key="user_id,isbn", batch_size=1000)

    return imported


@app.route("/")
def index():
    return render_template("index.html", popular_books=get_popular_books())


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    book_name = flask_request.values.get("book", "").strip()
    recommendations = []
    message = ""
    suggestions = []
    did_you_mean = ""
    debug_info = {}
    selected_book = None

    if book_name:
        recommendations, message, suggestions, did_you_mean, debug_info = recommend_books(book_name)
        selected_book = selected_book_from_debug(debug_info)

    return render_template(
        "recommend.html",
        book_name=book_name,
        selected_book=selected_book,
        recommendations=recommendations,
        message=message,
        suggestions=suggestions,
        did_you_mean=did_you_mean,
        debug_info=debug_info,
    )


@app.route("/api/suggestions")
def autocomplete_suggestions():
    query = flask_request.args.get("q", "").strip()
    return jsonify(fuzzy_title_matches(query, limit=8, cutoff=35))


@app.route("/book/<path:isbn>")
def book_details(isbn):
    row = find_book_row(isbn=isbn)

    if row is None:
        try:
            supabase_book = get_supabase_book(isbn)
        except RuntimeError:
            supabase_book = None

        if supabase_book:
            return render_template(
                "book_details.html",
                book=book_from_row(supabase_book),
                model_message="This book is in Supabase, but recommendations will be available after model retraining.",
            )

        flash("The requested book could not be found.", "warning")
        return redirect(url_for("index"))

    return render_template("book_details.html", book=book_from_row(row))


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if flask_request.method == "POST":
        name = flask_request.form.get("name", "").strip()
        email = flask_request.form.get("email", "").strip()
        message = flask_request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please complete every contact form field.", "warning")
            return redirect(url_for("contact"))

        try:
            store_contact_submission(name, email, message)
            flash("Thanks. Your message was submitted successfully.", "success")
        except RuntimeError as exc:
            flash(str(exc), "danger")

        return redirect(url_for("contact"))

    return render_template("contact.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if flask_request.method == "POST":
        username = flask_request.form.get("username", "").strip()
        password = flask_request.form.get("password", "")
        expected_username = os.environ.get("ADMIN_USERNAME", "admin")

        if username == expected_username and check_password_hash(admin_password_hash(), password):
            session.clear()
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Welcome back, admin.", "success")
            return redirect(flask_request.args.get("next") or url_for("admin_dashboard"))

        flash("Invalid admin credentials.", "danger")

    return render_template("admin_login.html")


@app.route("/admin/logout", methods=["POST"])
@admin_login_required
def admin_logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_login_required
def admin_dashboard():
    recent_books = []
    book_count = 0
    supabase_error = ""

    try:
        book_count = get_supabase_table_count("books")
        recent_books = get_supabase_books(order="created_at.desc", limit=10)
    except RuntimeError as exc:
        supabase_error = str(exc)

    return render_template(
        "admin_dashboard.html",
        recent_books=recent_books,
        book_count=book_count,
        model_count=len(MODEL_TITLES),
        supabase_error=supabase_error,
    )


@app.route("/admin/books")
@admin_login_required
def admin_books():
    query = flask_request.args.get("q", "").strip()
    page = max(1, flask_request.args.get("page", default=1, type=int))
    per_page = 50
    offset = (page - 1) * per_page
    books = []
    supabase_error = ""
    has_next = False

    try:
        books = search_supabase_books(query, limit=per_page + 1, offset=offset)
        has_next = len(books) > per_page
        books = books[:per_page]
    except RuntimeError as exc:
        supabase_error = str(exc)

    return render_template(
        "admin_books.html",
        books=books,
        query=query,
        page=page,
        has_next=has_next,
        supabase_error=supabase_error,
    )


@app.route("/admin/books/new", methods=["GET", "POST"])
@admin_login_required
def admin_book_new():
    if flask_request.method == "POST":
        try:
            upsert_supabase_book(flask_request.form)
            flash("Book saved to Supabase. Retrain the model to include it in recommendations.", "success")
            return redirect(url_for("admin_books"))
        except RuntimeError as exc:
            flash(str(exc), "danger")

    return render_template("admin_book_form.html", book={}, mode="new")


@app.route("/admin/books/<path:isbn>/edit", methods=["GET", "POST"])
@admin_login_required
def admin_book_edit(isbn):
    try:
        book = get_supabase_book(isbn)
    except RuntimeError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("admin_books"))

    if not book:
        flash("Book not found in Supabase.", "warning")
        return redirect(url_for("admin_books"))

    if flask_request.method == "POST":
        try:
            upsert_supabase_book(flask_request.form)
            flash("Book updated successfully.", "success")
            return redirect(url_for("admin_books"))
        except RuntimeError as exc:
            flash(str(exc), "danger")

    return render_template("admin_book_form.html", book=book, mode="edit")


@app.route("/admin/books/<path:isbn>/delete", methods=["POST"])
@admin_login_required
def admin_book_delete(isbn):
    try:
        delete_supabase_book(isbn)
        flash("Book deleted from Supabase.", "success")
    except RuntimeError as exc:
        flash(str(exc), "danger")

    return redirect(url_for("admin_books"))


@app.route("/admin/retrain", methods=["POST"])
@admin_login_required
def admin_retrain():
    try:
        progress = train_models_from_supabase()
        for item in progress:
            flash(item, "info")
        flash("Retraining completed successfully. The recommendation engine is now using the latest model files.", "success")
    except RuntimeError as exc:
        flash(str(exc), "danger")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/import-books", methods=["POST"])
@admin_login_required
def admin_import_books():
    try:
        count = import_current_books_to_supabase()
        flash(f"Imported or updated {count} books from the current model catalog into Supabase.", "success")
    except RuntimeError as exc:
        flash(str(exc), "danger")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/import-ratings", methods=["POST"])
@admin_login_required
def admin_import_ratings():
    try:
        count = import_dataset_ratings_to_supabase()
        flash(f"Imported or updated {count} ratings from Dataset/Ratings.csv into Supabase.", "success")
    except RuntimeError as exc:
        flash(str(exc), "danger")

    return redirect(url_for("admin_dashboard"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
