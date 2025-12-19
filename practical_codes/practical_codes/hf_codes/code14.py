from FlagEmbedding import BGEM3FlagModel

model=BGEM3FlagModel('BAAI/bge-m3',use_fp16=True)

sentence1="I love machine learning"

embeddings1=model.encode(sentence1,
                        batch_size=12,
                        max_length=8192,)['dense_vecs']

print(embeddings1)