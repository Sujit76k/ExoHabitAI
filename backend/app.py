from flask import Flask, jsonify
from flasgger import Swagger
from flask_cors import CORS
import logging

# ==============================
# 🚀 IMPORT BLUEPRINTS
# ==============================
from backend.routes.predict import predict_bp
from backend.routes.rank import rank_bp
from backend.routes.stats import stats_bp
from backend.routes.importance import importance_bp
from backend.routes.docs import docs_bp


# ==============================
# 🚀 CREATE APP
# ==============================
app = Flask(__name__)

# ==============================
# ⭐ BASIC CONFIG
# ==============================
app.config["SWAGGER"] = {
    "title": "ExoHabitAI API",
    "uiversion": 3,
    "description": "AI-powered Exoplanet Habitability Prediction System",
}

# ==============================
# ⭐ ENABLE EXTENSIONS
# ==============================
Swagger(app)

# Allow frontend dashboard access
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

# ==============================
# ⭐ LOGGING (Production Style)
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("exo-habit-ai")


# ==============================
# 🚀 REGISTER BLUEPRINTS
# ==============================
app.register_blueprint(predict_bp)
app.register_blueprint(rank_bp)
app.register_blueprint(stats_bp)
app.register_blueprint(importance_bp)
app.register_blueprint(docs_bp)


# ==============================
# 🧠 ROOT ROUTES
# ==============================
@app.route("/")
def home():
    return "🚀 ExoHabitAI API Running"


@app.route("/health")
def health():
    """
    Health check endpoint (Used in deployment monitoring)
    """
    return jsonify({"status": "ok"})


# ==============================
# 🚨 GLOBAL ERROR HANDLER
# ==============================
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled Error: {str(e)}")
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500


# ==============================
# ▶️ RUN SERVER
# ==============================
if __name__ == "__main__":
    logger.info("🚀 Starting ExoHabitAI API...")
    app.run(host="127.0.0.1", port=5000, debug=True)