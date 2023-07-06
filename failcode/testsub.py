from flask import Flask, request, jsonify
from flask_restx import Api, Resource


app = Flask(__name__)
api = Api(app)


@app.route("/client")
def HelloWorld():
    return "success"


if __name__ == "__main__":
    app.run(port=9966)
