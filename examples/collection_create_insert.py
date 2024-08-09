from pymilvus import MilvusClient, DataType, Collection
import time

client = MilvusClient(
    uri="http://localhost:19530"
)

def create_collection_and_get_state(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=5
    )

    res = client.get_load_state(
        collection_name=collection_name
    )

    print(res)


def create_schema_and_create_index():
# 3. Create a collection in customized setup mode
# 3.1. Create schema
    global schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

# 3.2. Add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)

    global index_params 
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )

    index_params.add_index(
        field_name="vector", 
        index_type="IVF_FLAT",
        metric_type="IP",
        params={ "nlist": 128 }
    )

    print(index_params)


def create_collection_and_get_state(collection_name):
    global schema
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    time.sleep(5)

    res = client.get_load_state(
        collection_name=collection_name
    )

    res = client.describe_collection(
        collection_name=collection_name
    )
    print(res)

def create_collection_and_index_separetlly(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    res = client.get_load_state(
        collection_name=collection_name
    )

    res = client.describe_collection(
        collection_name=collection_name
    )

    print(res)

def describe_collection(collection_name):
# 5. View Collections
    res = client.describe_collection(
        collection_name=collection_name
    )

    print(res)



def describe_index(collection_name,index_name):
# 5. Describe index
    res = client.list_indexes(
        collection_name=collection_name
    )

    print(res)
    res = client.describe_index(
        collection_name=collection_name,
        index_name=index_name
    )

    print(res)

def drop_index(collection_name,index_name):
    client.drop_index(
        collection_name=collection_name,
        index_name=index_name
    )

def get_all(collection_name):
    # Get all entities in a collection
    results = client.query(
        collection_name=collection_name,
        filter="",  # Empty string means no filter
        limit=10,
        output_fields=["*"]  # Get all fields
    )
    print(" == query == ")
    print(results)
    print(" == End == ")


def create_collection_and_insert_vectors(collection_name):

    c=client.create_collection(
        collection_name=collection_name,
        dimension=5,
        metric_type="IP"
    )

    data=[
        {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
        {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
        {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
        {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298"},
        {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794"},
        {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222"},
        {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392"},
        {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510"},
        {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381"},
        {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976"}
    ]

    res = client.insert(
        collection_name=collection_name,
        data=data
    )

    print(res)

    data=[
        {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294, -0.8648452746018911], "color": "black_9898"},
        {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281, 0.9660395877041965], "color": "red_7319"},
        {"id": 2, "vector": [-0.8864122635045097, 0.9260170474445351, 0.801326976181461, 0.6383943392381306, 0.7563037341572827], "color": "white_6465"},
        {"id": 3, "vector": [0.14594326235891586, -0.3775407299900644, -0.3765479013078812, 0.20612075380355122, 0.4902678929632145], "color": "orange_7580"},
        {"id": 4, "vector": [0.4548498669607359, -0.887610217681605, 0.5655081329910452, 0.19220509387904117, 0.016513983433433577], "color": "red_3314"},
        {"id": 5, "vector": [0.11755001847051827, -0.7295149788999611, 0.2608115847524266, -0.1719167007897875, 0.7417611743754855], "color": "black_9955"},
        {"id": 6, "vector": [0.9363032158314308, 0.030699901477745373, 0.8365910312319647, 0.7823840208444011, 0.2625222076909237], "color": "yellow_2461"},
        {"id": 7, "vector": [0.0754823906014721, -0.6390658668265143, 0.5610517334334937, -0.8986261118798251, 0.9372056764266794], "color": "white_5015"},
        {"id": 8, "vector": [-0.3038434006935904, 0.1279149203380523, 0.503958664270957, -0.2622661156746988, 0.7407627307791929], "color": "purple_6414"},
        {"id": 9, "vector": [-0.7125086947677588, -0.8050968321012257, -0.32608864121785786, 0.3255654958645424, 0.26227968923834233], "color": "brown_7231"}
    ]

    res = client.upsert(
        collection_name=collection_name,
        data=data
    )

    print(res)

    # Get all entities in a collection
    results = client.query(
        collection_name=collection_name,
        filter="1==1",  # Empty string means no filter
        limit=10,
        output_fields=["color"]  # Get all fields
    )
    print(" == query == ")
    print(results)
    print(" == End == ")

    res = client.get(
        collection_name=collection_name,
        ids=[0, 1, 2]
    )
    print("client.get")
    print(res)
    print("==========")

    #Delete entities according to id
    res = client.delete(
        collection_name=collection_name,
        filter="id in [4,5,6]"
    )

    print(res)



def drop_collection(collection_name):

    client.drop_collection(
        collection_name=collection_name
    )


###################################
#drop_collection("customized_setup_1")
print("=============")
#create_schema_and_create_index() ## the schema and index are stored into global variable
print("=============")
#create_collection_and_get_state("customized_setup_1")
print("=============")
#describe_index("customized_setup_1","vector")
print("=============")
#describe_collection("customized_setup_1")
print("=============")

drop_collection("collection_index_5_dim")
create_collection_and_insert_vectors("collection_index_5_dim")

