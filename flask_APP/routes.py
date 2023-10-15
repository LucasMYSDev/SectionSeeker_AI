"""Logged-in page routes."""
from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required, logout_user
import openpyxl
# Blueprint Configuration
main_bp = Blueprint(
    "main_bp", __name__, template_folder="templates", static_folder="static"
)

@main_bp.route("/", methods=["GET"])
@login_required
def dashboard():
    """Logged-in User Dashboard."""
    return render_template(
        "dashboard.jinja2",
        title="SectionSeekerAI",
        template="dashboard-template",
        current_user=current_user,
        body="You are now logged in!",
    )

@main_bp.route("/logout")
@login_required
def logout():
    """User log-out logic."""
    logout_user()
    return redirect(url_for("auth_bp.login"))

@main_bp.route('/contents')
def contents():
    return render_template('contents.jinja2')

@main_bp.route('/chat')
def chat():
    return render_template('chat.jinja2')









