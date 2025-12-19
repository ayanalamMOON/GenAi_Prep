from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

text ='''
Marwadi University (MU)[1] is a private university[2] located in Rajkot, Gujarat, India. It was established on 9 May 2016 by the Marwadi Education Foundation through The Gujarat Private Universities (Amendment) Act, 2016.[3] As of 2017, it offers 54 different courses.[4] It is graded A+ by NAAC.[5]

The university operates under the division of Marwadi Education Foundation's Group of Institutions (MEFGI). MEFGI commenced its operations in the year 2008. It was established as a primary unit of Marwadi Education Foundation under the Bombay Public Trust Act 1950. Marwadi University is aided by the Marwadi Shares and Finance Limited, a stock broking company in India and Chandarana Intermediaries Brokers Pvt. Ltd. (CIBPL), a firm dealing in technical and arbitrage trading.[6]

Campus
The campus is located on 52 acres of land, having a distance of nearly 40 minutes from railway and airports. The university comprises eight multi-storey buildings.[7] Laboratories, research facilities, student clubs, sports club and college cafeteria are available.[8]

There are two libraries with RFID technologies, 60+ computer systems, 50000+ books.[9] The campus also includes banking and ATM facilities.[10] Around 70+ buses function every day at regular intervals for students and staff.[11] There are hostel rooms with internet facilities, laundry, dance rooms, libraries etc. and capacity to occupy over 2000 students.[12]

'''

query="What is shoolini University?"

embedding_text = embedding_model.encode(text, batch_size=12, max_length=8192)['dense_vecs']

embedding_query = embedding_model.encode(query, batch_size=12, max_length=8192)['dense_vecs']

sim = cosine_similarity(
    [embedding_query],   # query embedding
    [embedding_text]       # document embedding
)[0][0]

print(f"Embedding vector length: {len(embedding_text)}")
print(embedding_text)
print("---------------------------------------------------------------------")
print(f"Embedding vector length: {len(embedding_query)}")
print(sim)