import os
import sys
import re
import time
import threading
import itertools
import textwrap
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# === CONFIGURATION ===
MODEL_PATH = "./model/Qwen3-1.7B-Q4_K_M.gguf"
N_CTX = 8196
N_THREADS = 12
N_GPU_LAYERS = 32
LOG_DIR = "./logs_apache"
MAX_LOG_LINES = 200
CHUNK_SIZE = 50
TOP_K_CHUNKS = 3

# === EMBEDDING MODEL ===
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_logs(logs_dict, chunk_size=CHUNK_SIZE):
    chunks = []
    mapping = []
    for fname, lines in logs_dict.items():
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i+chunk_size])
            chunks.append(chunk)
            mapping.append((fname, i))
    return chunks, mapping

def build_index_with_sklearn(chunks):
    embeddings = EMBEDDING_MODEL.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index = NearestNeighbors(n_neighbors=TOP_K_CHUNKS, metric='euclidean')
    index.fit(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(question, chunks, index, top_k=TOP_K_CHUNKS):
    q_embedding = EMBEDDING_MODEL.encode([question], convert_to_numpy=True, show_progress_bar=False)
    distances, indices = index.kneighbors(q_embedding, n_neighbors=top_k)
    return [chunks[i] for i in indices[0]]

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
    for fname in os.listdir(LOG_DIR):
        if fname.endswith(".log"):
            logs[fname] = read_logfile(fname)
    return logs

def extract_from_logs(question, logs):
    # Simplified extraction for "page(s) not found"
    if re.search(r"page[s]? .+not found", question, re.IGNORECASE):
        pages = set()
        for fname, lines in logs.items():
            if 'access' in fname or 'error' in fname:
                for l in lines:
                    m = re.search(r'404.*?"([^\s]+)"', l)
                    if m:
                        pages.add(m.group(1))
        return (", ".join(sorted(pages)) if pages else "(aucune page not found)", "Pages not found")
    return (None, None)

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

# Chargement modèle
print(f"⏳ Chargement du modèle depuis {MODEL_PATH} ...")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False
    )
    print("✅ Modèle chargé.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    sys.exit(1)

print("\n🧠 Pose ta question sur les logs Apache ('exit' pour quitter):\n")
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

    logs = get_all_logs()
    fact, fact_type = extract_from_logs(user_input, logs)

    if fact:
        prompt = (
            f"Expert Apache log assistant.\n"
            f"Question: {user_input}\n"
            f"Réponse extraite: {fact}\n"
            f"Formule une réponse claire et concise, sans information additionnelle.\nRéponse:"
        )
    else:
        chunks, _ = chunk_logs(logs)
        index, _ = build_index_with_sklearn(chunks)
        top_chunks = retrieve_relevant_chunks(user_input, chunks, index)

        context = "\n---\n".join(top_chunks)

        prompt = (
            f"Expert Apache log assistant.\n"
            f"Contexte extrait des logs:\n{context}\n\n"
            f"Question: {user_input}\n"
            f"Réponds uniquement en te basant sur les logs, sinon dis 'Not found in logs.'\nRéponse:"
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
            print("\n🤖 Qwen:")
            print(textwrap.fill(response, width=100))
            print()
    except Exception as e:
        spinner.stop()
        print(f"\n❌ Erreur durant l'inférence : {e}")
