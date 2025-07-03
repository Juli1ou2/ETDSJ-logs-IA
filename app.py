from flask import Flask, jsonify, render_template, request
from markupsafe import escape

from constants import CONSTANTS
from main import errors_500_present, find_best_question, getResponseFromModel, load_logs, load_questions, loadModel, newPrompt, pages_not_found_404, top_n_ips

app = Flask(__name__)
llm = loadModel()
questions, cleaned_questions, vectorizer, vecs = load_questions()

@app.route("/")
def index():
    return render_template('index.html', appName=CONSTANTS['APP_NAME'])

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    logs_lines = load_logs()
    idx = find_best_question(user_message, cleaned_questions, vectorizer, vecs)
    bot_response = ""
    
    if idx is not None:
        q = questions[idx]
        # Réponse rapide
        if "404" in q:
            bot_response = pages_not_found_404(logs_lines)
        elif "500" in q:
            bot_response = errors_500_present(logs_lines)
        elif "IP" in q:
            bot_response = str(top_n_ips(logs_lines))
        else:
            bot_response = "Analyse rapide non implémentée pour cette question, désolé."
            prompt = newPrompt(logs_lines, user_message)
            try:
                response = getResponseFromModel(llm, prompt)            
                if not response or len(response.replace("\n", "")) < 3:
                    bot_response += "\n" + "*réponse invalide*"
                else:
                    bot_response += "\n" + response.split("\n")[0]
            except Exception as e:
                print(f"\n❌ Erreur lors de la génération : {str(e)}")
                bot_response += f"\n❌ Erreur lors de la génération : {str(e)}"
    else:
        bot_response = "Je ne comprends pas la question. Reformule s'il-te-plaît."
        
    return jsonify({"response": bot_response})

@app.route("/logs", methods=["GET"])
def getLogs():
    logs_text = load_logs()
    return jsonify({"response": logs_text})