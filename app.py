from flask import Flask, jsonify, render_template, request
from markupsafe import escape

from constants import CONSTANTS
from main import getResponseFromModel, loadModel, newPrompt, read_logs

app = Flask(__name__)
llm = loadModel()

@app.route("/")
def index():
    return render_template('index.html', appName=CONSTANTS['APP_NAME'])

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    logs_text = read_logs()
    prompt = newPrompt(logs_text, user_message)
    bot_response = ""
    
    try:
        response = getResponseFromModel(llm, prompt)
        
        if not response or len(response.replace("\n", "")) < 3:
            bot_response = "*réponse invalide*"
        else:
            bot_response = response.split("\n")[0]
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération : {str(e)}")
        bot_response = f"\n❌ Erreur lors de la génération : {str(e)}"
    
    return jsonify({"response": bot_response})