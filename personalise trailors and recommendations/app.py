# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/outfits')
def outfits():
    return render_template('outfits.html')

@app.route('/ar')
def ar():
    return render_template('ar.html')

if __name__ == '__main__':
    app.run(debug=True)
