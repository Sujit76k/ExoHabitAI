from flask import Flask
from backend.routes.predict import predict_bp
from backend.routes.rank import rank_bp

app = Flask(__name__)

app.register_blueprint(predict_bp)
app.register_blueprint(rank_bp)

@app.route("/")
def home():
    return "ExoHabitAI API Running"

if __name__ == "__main__":
    app.run(debug=True)
