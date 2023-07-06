from flask import Flask, request, jsonify
from flask_restx import Api, Resource


app = Flask(__name__)
api = Api(app)


@app.route("/api")
def post():
    print("Hello")
    return "wtf"


if __name__ == "__main__":
    app.run(debug=True, port=9967)
