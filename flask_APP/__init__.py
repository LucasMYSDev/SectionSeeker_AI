"""Initialize app."""
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
import openai
db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    """Construct the core app object."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")

    app.config['UPLOAD_FOLDER'] = 'flask_APP/uploads'
    openai.api_key = app.config['OPENAI_API_KEY']  ## This is where API key is used!

    # Initialize Plugins
    db.init_app(app)
    login_manager.init_app(app)
    
    with app.app_context():
        from . import auth, routes, documents
        from .assets import compile_static_assets

        # Register Blueprints
        app.register_blueprint(routes.main_bp)
        app.register_blueprint(auth.auth_bp)
        app.register_blueprint(documents.document_bp)

        # Create Database Models
        db.create_all()

        # Compile static assets
        if app.config["FLASK_ENV"] == "development":
            compile_static_assets(app)

        return app

