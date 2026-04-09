"""Authentication routes — login / logout via TalentMatch backend."""

import os
import logging

import requests
from flask import Blueprint, render_template, request, redirect, url_for, session

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)

API_BASE = os.getenv("TALENTMATCH_API_URL", "http://localhost:5000")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("access_token"):
        return redirect(url_for("portal.dashboard"))

    error = None
    identifier = ""

    if request.method == "POST":
        identifier = request.form.get("email", "").strip()
        password   = request.form.get("password", "")

        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/auth/admin/login",
                json={"identifier": identifier, "password": password},
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                session["access_token"]  = data["access_token"]
                session["refresh_token"] = data["refresh_token"]
                session["user"]          = data.get("user", {})
                return redirect(url_for("portal.dashboard"))

            # Backend returned an error — show its message if available
            try:
                msg = resp.json().get("message") or resp.json().get("detail") or "Incorrect email or password."
            except Exception:
                msg = "Incorrect email or password."
            error = msg

        except requests.exceptions.ConnectionError:
            error = "Cannot reach the authentication server. Please try again later."
        except Exception as e:
            logger.error(f"auth login error: {e}")
            error = "Something went wrong. Please try again."

    return render_template("login.html", error=error, email=identifier)


@auth_bp.route("/logout")
def logout():
    access_token  = session.get("access_token")
    refresh_token = session.get("refresh_token")

    if refresh_token:
        try:
            requests.post(
                f"{API_BASE}/api/v1/auth/logout",
                json={"refresh_token": refresh_token, "access_token": access_token},
                timeout=5
            )
        except Exception:
            pass  # Best-effort — always clear local session

    session.clear()
    return redirect(url_for("auth.login"))
