from flask import Flask
from backend.routes.predict import predict_bp
from backend.routes.rank import rank_bp
from backend.routes.docs import docs_bp
from flasgger import Swagger
from flask_cors import CORS


app = Flask(__name__)

app.register_blueprint(predict_bp)
app.register_blueprint(rank_bp)
app.register_blueprint(docs_bp)
swagger = Swagger(app)
CORS(app)

@app.route("/")
def home():
    return "ExoHabitAI API Running"

if __name__ == "__main__":
    app.run(debug=True)
