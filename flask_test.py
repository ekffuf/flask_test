from flask import Flask

app = Flask(__name__)

result = 0.005
result2 = 0.35

@app.route('/')
def hello_world():
    if result or result2 >= 0.35:
        detect = 1
    else:
        detect = 0
    return str(detect)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
