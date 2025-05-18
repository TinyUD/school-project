from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    greeting = ""
    if request.method == 'POST':
        name = request.form.get('name')
        greeting = f"안녕하세요, {name}님!"
    return render_template('index.html', greeting=greeting)

if __name__ == '__main__':
    app.run(debug=True)
