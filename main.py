import threading
import itertools
import sys
import time
import os
from llama_cpp import Llama

# === CONFIGURATION ===
MODEL_PATH = "./model/Qwen3-1.7B-Q4_K_M.gguf"
N_CTX = 2048
N_THREADS = 8
LOG_DIR = "./logs_apache"
MAX_LOG_LINES = 100

# === UTILITAIRE POUR LIRE LES LOGS ===
def read_logs(max_lines=MAX_LOG_LINES):
    content = ""
    if not os.path.isdir(LOG_DIR):
        return "⚠️ Le dossier de logs Apache est introuvable."

    try:
        for file_name in os.listdir(LOG_DIR):
            if file_name.endswith(".log"):
                path = os.path.join(LOG_DIR, file_name)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        content += f"\n==== {file_name} ====\n" + "".join(lines[-max_lines:])
                except Exception as e:
                    content += f"\n⚠️ Impossible de lire {file_name} : {str(e)}\n"
    except Exception as e:
        return f"⚠️ Erreur lors de la lecture des fichiers de logs : {str(e)}"

    return content.strip() or "⚠️ Aucun contenu lisible trouvé dans les fichiers de logs."

# === INDICATEUR DE CHARGEMENT ===
class Spinner:
    def __init__(self, message="⏳ Le modèle réfléchit "):
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.stop_running = False
        self.message = message
        self.thread = threading.Thread(target=self.animate)

    def animate(self):
        while not self.stop_running:
            sys.stdout.write(f"\r{self.message}{next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()

# === CHARGEMENT DU MODÈLE ===
def loadModel():
    print("⏳ Chargement du modèle...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=32,
            verbose=False
        )
        print("✅ Modèle chargé.")
        return llm
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        sys.exit(1)
        
def newPrompt(logs_text, user_input):
    return (
        f"You are an expert assistant in Apache log analysis.\n"
        f"Here is a snippet of the logs:\n{logs_text}\n\n"
        f"Question: {user_input}\n"
        f"Answer clearly and concisely only.\n"
        f"Answer:"
    )
    
def getResponseFromModel(llm, prompt):
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.3
    )
    return output.get("choices", [{}])[0].get("text", "").strip()


# === BOUCLE PRINCIPALE ===
def runChat(llm):
    print("\n🧠 Entrez une question sur les logs Apache (tape 'exit' pour quitter):\n")
    while True:
        try:
            user_input = input("👤 Vous: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Fin de session.")
            break

        if not user_input:
            print("❗ Veuillez entrer une question.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("👋 À bientôt !")
            break

        logs_text = read_logs()

        # ✅ Nouveau prompt sans tokens spéciaux
        prompt = newPrompt(logs_text, user_input)

        spinner = Spinner()
        spinner.start()
        try:
            response = getResponseFromModel(llm, prompt)
            
            spinner.stop()
            if not response or len(response.replace("\n", "")) < 3:
                print("\n🤖 Qwen: (réponse vide ou invalide)\n")
            else:
                print(f"\n🤖 Qwen: {response}\n")

        except Exception as e:
            spinner.stop()
            print(f"\n❌ Erreur lors de la génération : {str(e)}\n")


def main():
    llm = loadModel()
    runChat(llm)
            
if __name__ == "__main__":
    main()