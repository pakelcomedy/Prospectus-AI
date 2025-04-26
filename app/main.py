from flask import Flask
import logging
from .routes import register_routes

def create_app() -> Flask:
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    register_routes(app)
    return app

if __name__ == "__main__":
    # Basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
    )
    app = create_app()
    # Listen on all interfaces, port 5000
    app.run(host="0.0.0.0", port=5000)
