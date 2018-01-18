#source \virtualenv\TFmetrics\bin\activate
from flask import Flask

app = Flask(__name__)

@app.route('/')
def import_network(path):





if __name__ == '__main__':
    app.run()
