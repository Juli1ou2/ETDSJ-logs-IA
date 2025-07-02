import threading
import itertools
import sys
import time
import os
import re
from llama_cpp import Llama

# === CONFIGURATION ===
MODEL_PATH = "./model/Qwen3-1.7B-Q4_K_M.gguf"
N_CTX = 8196
N_THREADS = 12
N_GPU_LAYERS = 32
LOG_DIR = "./logs_apache"
MAX_LOG_LINES = 200

# === UTILITAIRE POUR LIRE LES LOGS ===
def read_logs(max_lines=MAX_LOG_LINES, read_all=False):
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
                        # Toujours limiter à 200 lignes maximum, même si read_all=True
                        lines = lines[-MAX_LOG_LINES:]
                        content += f"\n==== {file_name} ====\n" + "".join(lines)
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
print(f"⏳ Chargement du modèle Qwen3-1.7B depuis {MODEL_PATH} ...")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False
    )
    print("✅ Modèle Qwen3-1.7B chargé.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {str(e)}")
    sys.exit(1)

def extract_fact_from_logs(question, logs_text):
    """
    Détecte les questions factuelles et extrait la réponse exacte des logs.
    Retourne (reponse_extraite, type_question) ou (None, None) si non reconnu.
    """
    # Pages sur lesquelles il y a un 'not found'
    if re.search(r"page[s]? .+not found", question, re.IGNORECASE):
        # Cherche dans access.log les URLs avec code 404
        pages = set()
        # Cherche dans tous les logs fournis
        for match in re.finditer(r'"[A-Z]+\s+([^\s]+)[^\"]*"\s+404\s', logs_text):
            pages.add(match.group(1))
        # Cherche dans error.log les chemins avec 'not found'
        for match in re.finditer(r'not found[^:]*: ([^\s]+)', logs_text, re.IGNORECASE):
            pages.add(match.group(1))
        return (", ".join(sorted(pages)) if pages else "(aucune page 'not found' trouvée)", "Pages avec not found")
    # IPs ayant accédé à une page spécifique
    m = re.search(r"ip[s]? .+acc(é|e)der? .+page ([^ ]+)", question, re.IGNORECASE)
    if m:
        page = m.group(2)
        pattern = re.compile(r'^(?P<ip>\S+).+?"[A-Z]+\s+' + re.escape(page) + r'(\s|\?|$).+?"', re.MULTILINE)
        ips = set(match.group("ip") for match in pattern.finditer(logs_text))
        return (", ".join(sorted(ips)) if ips else "(aucune IP trouvée)", f"IPs ayant accédé à {page}")
    # IPs ayant un code HTTP précis
    m = re.search(r"ip[s]? .+code http (\d{3})", question, re.IGNORECASE)
    if m:
        code = m.group(1)
        pattern = re.compile(r'^(?P<ip>\S+).+?"\s+' + re.escape(code) + r'\s', re.MULTILINE)
        ips = set(match.group("ip") for match in pattern.finditer(logs_text))
        return (", ".join(sorted(ips)) if ips else "(aucune IP trouvée)", f"IPs ayant code {code}")
    # URLs accédées par une IP
    m = re.search(r"url[s]? .+acc(é|e)der? .+ip ([0-9.]+)", question, re.IGNORECASE)
    if m:
        ip = m.group(2)
        pattern = re.compile(r'^' + re.escape(ip) + r'.+?"[A-Z]+\s+([^\s]+)', re.MULTILINE)
        urls = set(match.group(1) for match in pattern.finditer(logs_text))
        return (", ".join(sorted(urls)) if urls else "(aucune URL trouvée)", f"URLs accédées par {ip}")
    return (None, None)

# === BOUCLE PRINCIPALE ===
print("\n🧠 (Qwen3-1.7B) Entrez une question sur les logs Apache (tape 'exit' pour quitter):\n")
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

    # Pour l'extraction factuelle, on lit tous les logs (mais limité à 200 lignes par fichier)
    all_logs_text = read_logs(read_all=True)
    fact, fact_type = extract_fact_from_logs(user_input, all_logs_text)
    if fact is not None:
        # On demande au chatbot de reformuler la réponse exacte, sans inventer
        prompt = (
            f"You are an expert Apache log analysis assistant.\n"
            f"Here is a snippet of Apache logs:\n{all_logs_text}\n\n"
            f"User question: {user_input}\n"
            f"Exact answer extracted from the logs: {fact}\n"
            f"Reformulate the answer in a clear, concise, and direct way, without inventing or adding any information.\n"
            f"If the answer is a list, present it as a comma-separated list.\n"
            f"Answer:"
        )
    else:
        # Prompt optimisé pour la précision sur les logs Apache
        prompt = (
            f"You are an expert Apache log analysis assistant.\n"
            f"Here is a snippet of Apache logs:\n{logs_text}\n\n"
            f"User question: {user_input}\n"
            f"Answer as precisely and concisely as possible, based only on the provided logs. Do not invent or hallucinate information. If the answer is not present in the logs, say 'Not found in logs.'\n"
            f"If the answer is a list, present it as a comma-separated list.\n"
            f"Answer:"
        )

    spinner = Spinner()
    spinner.start()
    try:
        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.3,
            stop=["\n"]
        )
        spinner.stop()

        response = output.get("choices", [{}])[0].get("text", "").strip()
        response = next((line for line in response.splitlines() if line.strip()), "")

        if not response or len(response.replace("\n", "")) < 3:
            print("\n🤖 Qwen: (réponse vide ou invalide)\n")
        else:
            print(f"\n🤖 Qwen: {response}\n")

    except Exception as e:
        spinner.stop()
        print(f"\n❌ Erreur lors de la génération : {str(e)}\n")