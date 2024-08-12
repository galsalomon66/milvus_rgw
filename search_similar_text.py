from pymilvus import MilvusClient, DataType, Collection
import time
import random
import json

## using the following is the way milvus-lite is saving data locally
#client = MilvusClient("milvus_demo.db")


## connect to Milvus instance, it saves into S3(depends on configuration)
CLUSTER_ENDPOINT="http://localhost:19530"
client = MilvusClient(uri=CLUSTER_ENDPOINT)

if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
    print("collectiop dropped")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

# Load the collection into memory
client.load_collection("demo_collection")


## pip install milvus_model
from pymilvus import model

embedding_fn = model.DefaultEmbeddingFunction()

##Text strings to search from.
docs = [ "Artificial intelligence was founded as an academic discipline in 1956.", 
        "Alan Turing was the first person to conduct substantial research in AI.", 
        "Born in Maida Vale, London, Turing was raised in southern England.", ] 

vectors = embedding_fn.encode_documents(docs)
print("len(vectors) = ",len(vectors))

data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]

res = client.insert(collection_name="demo_collection", data=data) 
print(res)

##This code creates sample data with vectors and additional fields (text and subject), then inserts it into the collection.**Single Vector Search**
##Finally, let's perform a single vector search to find the top-K matching vectors

query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
    consistency_level="Strong" ## NOTE: without defining that, the search might return empty result.
)
print(res)

