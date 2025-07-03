import os
import sys
import time
import threading
import itertools
import textwrap
import faiss
import numpy as np
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import string

# === CONFIGURATION ===
MODEL_PATH = "./model/Qwen3-1.7B-Q4_K_M.gguf"
N_CTX = 8192
N_THREADS = 8
LOG_DIR = "./logs_apache"
MAX_LOG_LINES = 200
CHUNK_SIZE = 50
TOP_K_CHUNKS = 3

vectorizer = TfidfVectorizer()

def chunk_logs(logs_dict, chunk_size=CHUNK_SIZE):
    chunks = []
    for fname, lines in logs_dict.items():
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i+chunk_size])
            chunks.append(chunk)
    return chunks

def build_faiss_index(chunks):
    tfidf_matrix = vectorizer.fit_transform(chunks)
    embeddings = tfidf_matrix.toarray().astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(question, chunks, index, top_k=TOP_K_CHUNKS):
    q_vec = vectorizer.transform([question]).toarray().astype(np.float32)
    D, I = index.search(q_vec, top_k)
    return [chunks[i] for i in I[0]]

def read_logfile(filename, max_lines=MAX_LOG_LINES):
    path = os.path.join(LOG_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return lines[-max_lines:] if max_lines else lines
    except Exception:
        return []

def get_all_logs():
    logs = {}
    if not os.path.isdir(LOG_DIR):
        print(f"❌ Le dossier {LOG_DIR} n'existe pas.")
        return logs
    for fname in os.listdir(LOG_DIR):
        if fname.endswith(".log"):
            logs[fname] = read_logfile(fname)
    return logs

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

def main():
    print(f"⏳ Chargement du modèle depuis {MODEL_PATH} ...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False
        )
        print("✅ Modèle chargé.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    print("\n🧠 Pose ta question ('/log <question>' pour analyser les logs, 'exit' pour quitter):\n")

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

        if user_input.startswith("/log "):
            question = user_input[len("/log "):].strip()
            logs = get_all_logs()
            if not logs:
                print("❌ Aucun log disponible pour répondre.")
                continue
            chunks = chunk_logs(logs)
            if not chunks:
                print("❌ Les logs sont vides.")
                continue
            index = build_faiss_index(chunks)
            top_chunks = retrieve_relevant_chunks(question, chunks, index)
            context = "\n---\n".join(top_chunks)

            prompt = (
                f"Tu es un assistant expert qui analyse des logs Apache.\n"
                f"Voici des extraits pertinents des logs :\n\n{context}\n\n"
                f"Question : {question}\n\n"
                f"Réponds librement avec ta propre logique."
            )
            max_tokens = 256
            temperature = 0.5
        else:
            prompt = user_input
            max_tokens = 128
            temperature = 0.3

        spinner = Spinner()
        spinner.start()
        try:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            spinner.stop()

            response = output.get("choices", [{}])[0].get("text", "").strip()
            if not response:
                response = "Je ne trouve pas la réponse."

            print("\n🤖 Qwen:")
            print(textwrap.fill(response, width=100))
            print()
        except Exception as e:
            spinner.stop()
            print(f"\n❌ Erreur durant l'inférence : {e}")

# Chemin vers les logs Apache
LOGS_DIR = 'logs_apache/'

# Regex pour extraire une IP (IPv4 ou IPv6 simplifié)
IP_REGEX = re.compile(r'((?:\d{1,3}\.){3}\d{1,3}|[a-fA-F0-9:]+)')

def extract_ip(line):
    match = IP_REGEX.match(line.strip())
    if match:
        return match.group(1)
    return None

# Charger tous les logs dans une seule liste de lignes
log_lines = []
for filename in os.listdir(LOGS_DIR):
    if filename.endswith('.txt') or filename.endswith('.log'):
        with open(os.path.join(LOGS_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
            log_lines.extend(f.readlines())

# Fonctions d'analyse des logs

def count_total_accesses():
    return len(log_lines)

def count_status_code(code):
    count = 0
    for line in log_lines:
        match = re.search(r'" (\d{3}) ', line)
        if match and match.group(1) == str(code):
            count += 1
    return count

def most_active_ip():
    ips = [extract_ip(line) for line in log_lines if extract_ip(line)]
    if not ips:
        return None
    counter = Counter(ips)
    return counter.most_common(1)[0]

def unique_ips():
    ips = set(extract_ip(line) for line in log_lines if extract_ip(line))
    return len(ips)

def top_n_ips(n=5):
    ips = [extract_ip(line) for line in log_lines if extract_ip(line)]
    counter = Counter(ips)
    return counter.most_common(n)

# === MAPPING QUESTIONS.TXT ===
QUESTIONS_MAP = {
    "Quelles sont les pages \"not found\" (erreurs 404) ?": lambda: pages_not_found_404(),
    "Y a-t-il des erreurs 500 dans les logs ?": lambda: errors_500_present(),
    "Quels sont les IPs qui font le plus de requêtes ?": lambda: top_n_ips(5),
    "Quels sont les URL les plus demandés ?": lambda: most_requested_urls(5),
    "Y a-t-il des tentatives d'accès suspectes ou erreurs d'authentification ?": lambda: suspicious_access_or_auth_errors(),
    "Quels sont les pics d'activité sur le serveur ?": lambda: activity_peaks(),
    "Combien de requêtes par minute/heures ?": lambda: requests_per_time(),
    "Quelle est la taille moyenne des réponses ?": lambda: average_response_size(),
    "Y a-t-il des erreurs de timeout ?": lambda: timeout_errors_present(),
    "Quel est le temps moyen de réponse ?": lambda: average_response_time(),
    "Y a-t-il des redirections (301, 302) fréquentes ?": lambda: frequent_redirections(),
    "Liste-moi les requêtes avec des codes d'état HTTP différents de 200.": lambda: non_200_requests(),
}

# === FONCTIONS D'ANALYSE POUR CHAQUE QUESTION ===
def pages_not_found_404():
    urls = []
    for line in log_lines:
        match = re.search(r'"(GET|POST|HEAD|PUT|DELETE|OPTIONS) ([^ ]+) [^"]+" 404', line)
        if match:
            urls.append(match.group(2))
    if not urls:
        return "Aucune page 404 trouvée."
    return "Pages 'not found' (404) :\n" + '\n'.join(set(urls))

def errors_500_present():
    count = count_status_code(500)
    return f"Il y a {count} erreur(s) 500 dans les logs." if count else "Aucune erreur 500 trouvée."

def most_requested_urls(n=5):
    urls = []
    for line in log_lines:
        match = re.search(r'"(GET|POST|HEAD|PUT|DELETE|OPTIONS) ([^ ]+) [^"]+" \d{3}', line)
        if match:
            urls.append(match.group(2))
    if not urls:
        return "Aucune URL trouvée."
    counter = Counter(urls)
    return "URLs les plus demandées :\n" + '\n'.join([f"{url} ({count} requêtes)" for url, count in counter.most_common(n)])

def suspicious_access_or_auth_errors():
    suspicious = []
    for line in log_lines:
        if re.search(r'401|403|denied|unauthorized|forbidden|login|auth', line, re.IGNORECASE):
            suspicious.append(line.strip())
    if not suspicious:
        return "Aucune tentative suspecte ou erreur d'authentification trouvée."
    return f"{len(suspicious)} accès suspects ou erreurs d'authentification détectés."

def activity_peaks():
    # Placeholder simple : à améliorer pour détecter de vrais pics
    return "Détection de pics d'activité non implémentée (à coder selon la granularité temporelle des logs)."

def requests_per_time():
    # Placeholder simple : à améliorer pour calculer par minute/heure
    return "Calcul du nombre de requêtes par minute/heure non implémenté (à coder selon le format de date des logs)."

def average_response_size():
    sizes = []
    for line in log_lines:
        match = re.search(r'" \d{3} (\d+)', line)
        if match:
            sizes.append(int(match.group(1)))
    if not sizes:
        return "Impossible de calculer la taille moyenne des réponses."
    avg = sum(sizes) / len(sizes)
    return f"Taille moyenne des réponses : {avg:.2f} octets."

def timeout_errors_present():
    count = sum(1 for line in log_lines if 'timeout' in line.lower())
    return f"{count} erreur(s) de timeout détectée(s)." if count else "Aucune erreur de timeout trouvée."

def average_response_time():
    # Placeholder : à adapter selon la présence du temps de réponse dans les logs
    return "Calcul du temps moyen de réponse non implémenté (à coder selon le format des logs)."

def frequent_redirections():
    count_301 = count_status_code(301)
    count_302 = count_status_code(302)
    total = count_301 + count_302
    if total == 0:
        return "Aucune redirection 301 ou 302 trouvée."
    return f"Redirections fréquentes : {count_301} (301), {count_302} (302)"

def non_200_requests():
    reqs = []
    for line in log_lines:
        match = re.search(r'"(GET|POST|HEAD|PUT|DELETE|OPTIONS) ([^ ]+) [^"]+" (?!200)\d{3}', line)
        if match:
            reqs.append(f"{match.group(1)} {match.group(2)}")
    if not reqs:
        return "Toutes les requêtes ont un code 200."
    return "Requêtes avec code différent de 200 :\n" + '\n'.join(reqs)

# Charger les questions de questions.txt
with open('questions.txt', encoding='utf-8') as f:
    QUESTIONS_LIST = [q.strip() for q in f if q.strip()]

# Vectoriser les questions
vectorizer_q = TfidfVectorizer().fit(QUESTIONS_LIST)
questions_vecs = vectorizer_q.transform(QUESTIONS_LIST)

SIMILARITY_THRESHOLD = 0.4  # Plus tolérant
FUZZY_THRESHOLD = 0.7       # Pour le ratio difflib

# Fonction de nettoyage avancé de texte
def clean_text(text):
    text = text.strip().lower()
    # Normalisation unicode
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Remplacement apostrophes et guillemets
    text = text.replace('’', "'").replace('"', '"').replace('"', '"')
    # Remplacement espaces insécables
    text = text.replace('\xa0', ' ').replace('\u202f', ' ')
    # Suppression ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Suppression espaces multiples
    text = ' '.join(text.split())
    return text

# Nettoyer les questions pour le matching
CLEANED_QUESTIONS_LIST = [clean_text(q) for q in QUESTIONS_LIST]

# Adapter le vectorizer sur les questions nettoyées
vectorizer_q = TfidfVectorizer().fit(CLEANED_QUESTIONS_LIST)
questions_vecs = vectorizer_q.transform(CLEANED_QUESTIONS_LIST)

def find_best_question(user_question):
    cleaned = clean_text(user_question)
    user_vec = vectorizer_q.transform([cleaned])
    sims = cosine_similarity(user_vec, questions_vecs)[0]
    best_idx = sims.argmax()
    if sims[best_idx] >= SIMILARITY_THRESHOLD:
        return QUESTIONS_LIST[best_idx], best_idx, sims[best_idx]
    # Fallback : fuzzy matching
    ratios = [difflib.SequenceMatcher(None, cleaned, cq).ratio() for cq in CLEANED_QUESTIONS_LIST]
    fuzzy_idx = max(range(len(ratios)), key=lambda i: ratios[i])
    if ratios[fuzzy_idx] >= FUZZY_THRESHOLD:
        return QUESTIONS_LIST[fuzzy_idx], fuzzy_idx, ratios[fuzzy_idx]
    return None, None, max(sims[best_idx], ratios[fuzzy_idx])

# === NOUVELLE BOUCLE CLI ===
if __name__ == "__main__":
    print("Chatbot Apache prêt ! Tapez 'quit' pour arrêter.")
    print("Vous pouvez poser les questions suivantes (ou des variantes proches) :")
    questions_to_hide = [
        "Y a-t-il des tentatives d'accès suspectes ou erreurs d'authentification ?",
        "Liste-moi les requêtes avec des codes d'état HTTP différents de 200.",
        "Quels sont les pics d'activité sur le serveur ?",
        "Combien de requêtes par minute/heures ?"
    ]
    for q in QUESTIONS_MAP:
        if q not in questions_to_hide:
            print(f"- {q}")
    last_suggestion_idx = None
    while True:
        message = input("Vous : ").strip()
        if message.lower() == 'quit':
            break
        if message.lower() in ['oui', 'yes', 'ok'] and last_suggestion_idx is not None:
            suggestion = QUESTIONS_LIST[last_suggestion_idx]
            print(f"(J'exécute la question suggérée : '{suggestion}')")
            response = QUESTIONS_MAP[suggestion]()
            last_suggestion_idx = None
        else:
            best_q, best_idx, score = find_best_question(message)
            if best_q and best_q in QUESTIONS_MAP:
                print(f"(J'ai compris : '{best_q}')")
                response = QUESTIONS_MAP[best_q]()
                last_suggestion_idx = None
            else:
                # Proposer la question la plus proche trouvée (par index)
                if questions_vecs.shape[0] > 0:
                    cleaned = clean_text(message)
                    user_vec = vectorizer_q.transform([cleaned])
                    sims = cosine_similarity(user_vec, questions_vecs)[0]
                    best_idx = sims.argmax()
                    suggestion = QUESTIONS_LIST[best_idx]
                    response = (
                        "Je ne comprends pas bien la question. "
                        f"Voulais-tu dire : '{suggestion}' ?\n"
                        "(Réponds 'oui' pour que j'exécute cette question, ou reformule ta demande.)"
                    )
                    last_suggestion_idx = best_idx
                else:
                    response = "Je ne comprends que les questions du fichier questions.txt ou des variantes proches."
                    last_suggestion_idx = None
        print(f"Bot : {response}")
