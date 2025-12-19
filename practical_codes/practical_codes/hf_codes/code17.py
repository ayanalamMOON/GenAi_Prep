# ============================================================
# RAG PIPELINE WITH EXTERNAL MOVIE DATASET — MODULAR VERSION
# ============================================================

from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ------------------------------------------------------------
# FUNCTION 1: Load Movie Dataset
# ------------------------------------------------------------
def load_movie_dataset():
    return [
        "Shadow Hunters: ACTION FANTASY THRILLER",
        "Eternal Bonds: DRAMA ROMANCE FAMILY",
        "Neon Horizon: SCI-FI ACTION ADVENTURE",
        "Whispers in the Dark: HORROR MYSTERY THRILLER",
        "Crimson Crown: HISTORICAL WAR DRAMA",
        "Starlight Dreams: ROMANCE COMEDY MUSIC",
        "Code Breakers: TECH THRILLER CRIME",
        "The Lost Kingdom: ADVENTURE ACTION FANTASY",
        "Broken Silence: CRIME DRAMA SUSPENSE",
        "Frozen Ashes: SURVIVAL DRAMA ACTION"
    ]

# ------------------------------------------------------------
# FUNCTION 2: Load Embedding Model + Generate Embeddings
# ------------------------------------------------------------
def embed_movies(movies_list):
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    embeddings = [
        model.encode(movie, batch_size=12, max_length=8192)["dense_vecs"]
        for movie in movies_list
    ]
    return model, embeddings

# ------------------------------------------------------------
# FUNCTION 3: Retrieve Top-K Relevant Movies
# ------------------------------------------------------------
def retrieve_top_movies(query, model, movies_list, movie_embeddings, top_k=4):
    query_emb = model.encode(query, batch_size=12, max_length=8192)["dense_vecs"]

    similarity_scores = {}
    for i, emb in enumerate(movie_embeddings):
        sim = cosine_similarity([query_emb], [emb])[0][0]
        similarity_scores[movies_list[i]] = sim

    sorted_results = sorted(
        similarity_scores.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_results[:top_k]

# ------------------------------------------------------------
# FUNCTION 4: Build the RAG Prompt
# ------------------------------------------------------------
def build_rag_prompt(query, retrieved_movies):
    context_text = "\n".join([f"- {m[0]}" for m in retrieved_movies])
    
    prompt = f"""
You are a movie recommendation expert.

USER QUERY:
{query}

RETRIEVED MOVIES (Context Provided):
{context_text}

TASK:
Using ONLY the above movies as your context, recommend movies to the user.
Clearly explain why each recommended movie matches the user's taste.
Do NOT mention similarity scores or embeddings.
"""
    return prompt

# ------------------------------------------------------------
# FUNCTION 5: Invoke HF LLM
# ------------------------------------------------------------
def generate_llm_response(prompt):
    pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    
    result = pipe(prompt, max_new_tokens=200, temperature=0.7, num_return_sequences=1)
    return result[0]['generated_text']

# ------------------------------------------------------------
# FUNCTION 6: Full RAG Pipeline Executor
# ------------------------------------------------------------
def run_rag_pipeline():
    # Load dataset
    movies_list = load_movie_dataset()

    # Embeddings
    model, movie_embeddings = embed_movies(movies_list)

    # User Input
    query = input("Enter movie genre or mood: ")

    # Retrieve
    retrieved = retrieve_top_movies(query, model, movies_list, movie_embeddings)

    # Build prompt
    prompt = build_rag_prompt(query, retrieved)

    # LLM response
    answer = generate_llm_response(prompt)

    print("\n====================== RAG ANSWER ======================")
    print(answer)

# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------
run_rag_pipeline()