"""Create and bundle CSS and JS files."""
from flask_assets import Bundle, Environment


def compile_static_assets(app):
    """Configure static asset bundles."""
    assets = Environment(app)
    Environment.auto_build = True
    Environment.debug = False
    # Stylesheets Bundles
    account_less_bundle = Bundle(
        "src/less/account.less",
        filters="less,cssmin",
        output="dist/css/account.css",
        extra={"rel": "stylesheet/less"},
    )
    dashboard_less_bundle = Bundle(
        "src/less/dashboard.less",
        filters="less,cssmin",
        output="dist/css/dashboard.css",
        extra={"rel": "stylesheet/less"},
    )

    previous_main_css_bundle = Bundle(
        "static/dist/css/previous_main.css",
        output="dist/css/previous_main.css"
    )

    # JavaScript Bundle
    previous_main_js_bundle = Bundle(
        "static/js/previous_main.js",
        output="dist/js/previous_main.min.js"
    )

    
    # JavaScript Bundle
    js_bundle = Bundle("src/js/main.js", filters="jsmin", output="dist/js/main.min.js")
    # Register assets
    assets.register("account_less_bundle", account_less_bundle)
    assets.register("dashboard_less_bundle", dashboard_less_bundle)
    assets.register("js_all", js_bundle)
    assets.register("previous_main_css_bundle", previous_main_css_bundle)
    assets.register("previous_main_js_bundle", previous_main_js_bundle)
    # Build assets
    account_less_bundle.build()
    dashboard_less_bundle.build()
    js_bundle.build()
    previous_main_css_bundle.build()
    previous_main_js_bundle.build()
