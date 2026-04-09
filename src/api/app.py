"""Flask application factory."""

import os

from flask import Flask
from flask_cors import CORS

from src.config.logging_config import setup_logging
from src.config.settings import get_settings


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask application
    """
    settings = get_settings()

    # Setup logging
    setup_logging()

    # Get template folder path
    template_dir = os.path.join(os.path.dirname(__file__), "templates")

    # Create Flask app with templates
    app = Flask(__name__, template_folder=template_dir)

    # Configure app
    app.config["DEBUG"] = settings.flask_debug
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    from src.api.routes.auth import auth_bp
    from src.api.routes.health import health_bp
    from src.api.routes.screening import screening_bp
    from src.api.routes.data_sync import data_sync_bp
    from src.api.routes.rubrics import rubrics_bp
    from src.api.routes.portal import portal_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(screening_bp, url_prefix="/api/screening")
    app.register_blueprint(data_sync_bp, url_prefix="/api")
    app.register_blueprint(rubrics_bp, url_prefix="/api")
    app.register_blueprint(portal_bp)  # Portal at root for web interface

    # Redirect unauthenticated / expired-token requests to /login
    import requests as http
    from flask import request as req, redirect, url_for, session

    OPEN_ROUTES = {"/login", "/health"}
    API_BASE = os.environ.get("TALENTMATCH_API_URL", "http://localhost:5000")

    @app.before_request
    def require_login():
        if req.path in OPEN_ROUTES or req.path.startswith("/api/"):
            return None

        token = session.get("access_token")
        if not token:
            return redirect(url_for("auth.login"))

        # Verify token is still valid with the backend
        try:
            resp = http.post(
                f"{API_BASE}/api/v1/auth/token/verify",
                json={"token": token},
                timeout=5
            )
            if resp.status_code != 200 or not resp.json().get("valid"):
                session.clear()
                return redirect(url_for("auth.login"))
        except Exception:
            # If verify call fails (network), let them through — don't lock out on infra issues
            pass

    # Register error handlers
    register_error_handlers(app)

    return app


def register_error_handlers(app: Flask) -> None:
    """Register global error handlers.

    Args:
        app: Flask application
    """
    from flask import jsonify

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad Request",
            "message": str(error.description),
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested resource was not found",
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
        }), 500
