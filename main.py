from collections import Counter
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import threading
import itertools
import sys
import time
import os
import unicodedata
from llama_cpp import Llama

# === CONFIGURATION ===
MODEL_PATH = "./model/Qwen3-1.7B-Q4_K_M.gguf"
N_CTX = 2048
N_THREADS = 8
LOGS_DIR = "./logs_apache"
QUESTIONS_FILE = 'questions.txt'
MAX_LOG_LINES = 100
IP_REGEX = re.compile(r'((?:\d{1,3}\.){3}\d{1,3}|[a-fA-F0-9:]+)')
SIMILARITY_THRESHOLD = 0.4
FUZZY_THRESHOLD = 0.7

def clean_text(text):
    text = text.strip().lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# === UTILITAIRE POUR LIRE LES LOGS ===
def load_logs():
    log_lines = []
    if not os.path.isdir(LOGS_DIR):
        return "⚠️ Le dossier de logs Apache est introuvable."
    try:
        for filename in os.listdir(LOGS_DIR):
            if filename.endswith('.txt') or filename.endswith('.log'):
                with open(os.path.join(LOGS_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    log_lines.extend(f.readlines())
        return log_lines
    except Exception as e:
        return f"⚠️ Erreur lors de la lecture des fichiers de logs : {str(e)}"

# === UTILITAIRE POUR LIRE LES QUESTIONS ===
def load_questions():
    with open(QUESTIONS_FILE, encoding='utf-8') as f:
        questions = [q.strip() for q in f if q.strip()]
    cleaned = [clean_text(q) for q in questions]
    vectorizer = TfidfVectorizer().fit(cleaned)
    vecs = vectorizer.transform(cleaned)
    return questions, cleaned, vectorizer, vecs

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
        
def extract_ip(line):
    match = re.match(r'((?:\d{1,3}\.){3}\d{1,3}|[a-fA-F0-9:]+)', line.strip())
    return match.group(1) if match else None

def count_status_code(log_lines, code):
    return sum(1 for line in log_lines if re.search(r'" (\d{3}) ', line) and re.search(r'" (\d{3}) ', line).group(1) == str(code))

def pages_not_found_404(log_lines):
    urls = [re.search(r'"(GET|POST|HEAD|PUT|DELETE|OPTIONS) ([^ ]+) [^"]+" 404', line).group(2)
            for line in log_lines if re.search(r'"(GET|POST|HEAD|PUT|DELETE|OPTIONS) ([^ ]+) [^"]+" 404', line)]
    return "Aucune page 404 trouvée." if not urls else "Pages 'not found' (404) :\n" + '\n'.join(set(urls))

def errors_500_present(log_lines):
    count = count_status_code(log_lines, 500)
    return f"Il y a {count} erreur(s) 500 dans les logs." if count else "Aucune erreur 500 trouvée."

def top_n_ips(log_lines, n=5):
    ips = [extract_ip(line) for line in log_lines if extract_ip(line)]
    counter = Counter(ips)
    return counter.most_common(n)

def find_best_question(user_question, cleaned_questions, vectorizer, vecs):
    cleaned = clean_text(user_question)
    user_vec = vectorizer.transform([cleaned])
    sims = cosine_similarity(user_vec, vecs)[0]
    best_idx = sims.argmax()
    if sims[best_idx] >= SIMILARITY_THRESHOLD:
        return best_idx
    # Fallback fuzzy
    ratios = [difflib.SequenceMatcher(None, cleaned, cq).ratio() for cq in cleaned_questions]
    fuzzy_idx = max(range(len(ratios)), key=lambda i: ratios[i])
    if ratios[fuzzy_idx] >= FUZZY_THRESHOLD:
        return fuzzy_idx
    return None

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