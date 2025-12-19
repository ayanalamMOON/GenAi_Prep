# !pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


movies_list = [
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

# Convert movie list into embeddings
embeddings_list = []
for movie in movies_list:
    emb = model.encode(
        movie,
        batch_size=12,
        max_length=8192
    )['dense_vecs']
    embeddings_list.append(emb)


query = input("ENTER THE GENRE FOR WHICH YOU WANT TO SEE MOVIES: ")


query_embedding = model.encode(
    query,
    batch_size=12,
    max_length=8192
)['dense_vecs']


similarity_scores = {}

for i, movie_emb in enumerate(embeddings_list):
    similarity = cosine_similarity(
        [query_embedding],
        [movie_emb]
    )[0][0]

    similarity_scores[movies_list[i]] = similarity


sorted_movies = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

#Retrived data 
print("\n🎬 Top Recommended Movies:")
for movie, score in sorted_movies[:4]:
    print(f"{movie}   ---> Similarity: {score:.4f}")
