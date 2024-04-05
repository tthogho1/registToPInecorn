from flask import Flask, render_template,request,send_from_directory,make_response
from flask_cors import CORS
from searchVector import search_module
from waitress import serve

app = Flask(__name__)
CORS(app)

# for metal test site
app.register_blueprint(search_module)

@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/testb', methods=['POST'])
#def test_post():
#    prompt = request.form.get('name')
#    print(f"prompt : {prompt}")
#    return 'MetalPost'

if __name__ == "__main__":
    #app.run(debug=True)
    serve(app, host="0.0.0.0", port=5000)