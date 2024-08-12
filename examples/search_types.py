## based on: https://milvus.io/docs/single-vector-search.md
from pymilvus import MilvusClient, DataType, Collection
import time
import random
import json

CLUSTER_ENDPOINT="http://localhost:19530"

client = MilvusClient(
    uri=CLUSTER_ENDPOINT
#    token=TOKEN 
)

COLLECTION_NAME="quick_setup"
colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
input_data = []

def drop_collection(collection_name):
    client.drop_collection(
        collection_name=collection_name
    )

def create_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=5,
        metric_type="IP"
    )

def insert_data(collection_name):
    global input_data
    res = client.insert(
        collection_name=collection_name,
        data=input_data
    )
    input_data=[]
    print(res)
    
def create_data():
    for i in range(1000):
        current_color = random.choice(colors)
        input_data.append({
            "id": i,
            "vector": [ random.uniform(-1, 1) for _ in range(5) ],
            "color": current_color,
            "color_tag": f"{current_color}_{str(random.randint(1000, 9999))}"
        })
    insert_data(COLLECTION_NAME)

def create_partition(partition_name):
    client.create_partition(
        collection_name=COLLECTION_NAME,
        partition_name=partition_name
    )

#############################
#####################   start 
#############################
input_data = []
red_data = [ {"id": i, "vector": [ random.uniform(-1, 1) for _ in range(5) ], "color": "red", "color_tag": f"red_{str(random.randint(1000, 9999))}" } for i in range(500) ]
blue_data = [ {"id": i, "vector": [ random.uniform(-1, 1) for _ in range(5) ], "color": "blue", "color_tag": f"blue_{str(random.randint(1000, 9999))}" } for i in range(500) ]

drop_collection(COLLECTION_NAME)
create_collection(COLLECTION_NAME)
create_data()

input_data=[]
single_vector_for_insert = query_vector
input_data.append({
        "id": 2661616,
        "vector": single_vector_for_insert,
        "color": "red",
        "color_tag": "red_8877"
})
insert_data(COLLECTION_NAME)

create_partition("red")
create_partition("blue")

## insert single vector into partition
red_data.append({
        "id": 123456,
        "vector": query_vector,
        "color": "red",
        "color_tag": "red_123"
})
res = client.insert(
    collection_name=COLLECTION_NAME,
    data=red_data,
    partition_name="red"
)
print(res)

res = client.insert(
    collection_name=COLLECTION_NAME,
    data=blue_data,
    partition_name="blue"
)
print(res)

######################  Single-vector search

## TODO return zero result ## L2, IP, COSINE, JACCARD, HAMMING.
res = client.search(
    collection_name=COLLECTION_NAME, 
    # Replace with your query vector
    #data=[{"id": 12345, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "red", "color_tag": "123"} ],
    data=[query_vector],
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

print("search Single-vector ")
result = json.dumps(res, indent=4)
print(result)


#################### bulk search
## NOTE: return zero result
res = client.search(
    collection_name=COLLECTION_NAME, 
    data=[
        [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
        [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]
    ], # Replace with your query vectors
    limit=2, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

print("bulk search")
result = json.dumps(res, indent=4)
print(result)


#################### partition search

## NOTE: return zero result
res = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=5,
    search_params={"metric_type": "IP", "params": {"level": 1}},
    partition_names=["red"]
)

print("partition search")
print(res)

#################### Search with output fields

res = client.search(
    collection_name=COLLECTION_NAME, 
    data=[query_vector], 
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}}, # Search parameters
    output_fields=["color"] # Output fields to return
)

print("output fields search")
result = json.dumps(res, indent=4)
print(result)

#################### filter search
res = client.search(
    collection_name=COLLECTION_NAME, 
    data=[query_vector],
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}}, # Search parameters
    output_fields=["color"], # Output fields to return
    filter='color like "red%"'
)

print("filter search")
result = json.dumps(res, indent=4)
print(result)

##############  range search
search_params = {
    "metric_type": "IP",
    "params": {
        "radius": 0.8, # Radius of the search circle
        "range_filter": 1.0 # Range filter to filter out vectors that are not within the search circle
    }
}

res = client.search(
    collection_name=COLLECTION_NAME, # Replace with the actual name of your collection
    data=[query_vector],
    limit=3, # Max. number of search results to return
    search_params=search_params, # Search parameters
    output_fields=["color"], # Output fields to return
)

print("range search")
result = json.dumps(res, indent=4)
print(result)

############### group search
#client = MilvusClient(uri='http://localhost:19530') # Milvus server address

#client.load_collection("quick_setup") # Collection name

exit(0)
res = client.search(
    collection_name=COLLECTION_NAME, # Collection name
    #data=[[0.14529211512077012, 0.9147257273453546, 0.7965055218724449, 0.7009258593102812, 0.5605206522382088]], # Query vector
    data=[query_vector],
    search_params={
    "metric_type": "IP",
    "params": {"nprobe": 10},
    }, # Search parameters
    limit=10, # Max. number of search results to return
    group_by_field="color", # Group results by document ID
    output_fields=["color", "color_tag"]
)

doc_ids = [result['entity']['color'] for result in res[0]]

print("group search")
print(doc_ids)

