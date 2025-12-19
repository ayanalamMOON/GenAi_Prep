#cosine similarity


# !pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

model=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)

sentence1="I am going to school"

embeddings1=model.encode(sentence1,
                        batch_size=12,
                        max_length=8192,)['dense_vecs']

print(embeddings1)

sentence2="I am going to home"

embeddings2=model.encode(sentence2,
                        batch_size=12,
                        max_length=8192,)['dense_vecs']

print(embeddings2)

similiarity_list= cosine_similarity(
      [embeddings1], 
      [embeddings2]      
  )[0][0]

print(similarity)


