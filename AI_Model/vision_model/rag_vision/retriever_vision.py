from typing import Dict
import os
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables from .env file - specify path explicitly
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
    
class doc_retriver:
    def retriver(self, client, query: str, collection_name: str = "DogDisease", property_name: str = "disease_name"):
        from weaviate.collections.classes.filters import Filter
        collection = client.collections.get(collection_name)
        results = collection.query.fetch_objects(
            filters=Filter.by_property(property_name).equal(query)
        )
        
        return [obj.properties for obj in results.objects]

class MetaDataStore:
    def get_client(self):
        import weaviate
        from weaviate.auth import AuthApiKey
        
        WEAVIATE_URL = os.getenv('WEAVIATE_URL')
        WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
        
        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set")
        
        auth = AuthApiKey(api_key=WEAVIATE_API_KEY)
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.AuthApiKey(WEAVIATE_API_KEY),
        )
        return client
    
    def create_collection(self, client, collection_name:str):
        import weaviate.classes as wvc
        from weaviate.classes.config import Property, DataType
        
        # Check if the collection already exists and delete it if it does
        if client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' already exists. Deleting and recreating it.")
            client.collections.delete(collection_name)
        
        # Define the vectorizer configuration, using 'text2vec-huggingface' as a common default.
        # If this vectorizer is not enabled on your Weaviate instance, you might need to change it
        # to another available vectorizer (e.g., text2vec-openai, text2vec-contextionary) or .none()
        
        
        # Create the collection with the vectorizer
        client.collections.create(
            name=collection_name,
            vectorizer_config=None,
            description="Dog disease metadata for RAG medical system",
            properties=[
                Property(name="disease_name", data_type=DataType.TEXT),
                Property(name="possible_condition", data_type=DataType.TEXT),
                Property(name="severity_level", data_type=DataType.TEXT),
                Property(name="first_aid", data_type=DataType.TEXT),
                Property(name="otc_recommendations", data_type=DataType.TEXT),
                Property(name="monitoring", data_type=DataType.TEXT),
                Property(name="vet_visit_urgency", data_type=DataType.TEXT),
                Property(name="emergency_flag", data_type=DataType.BOOL),
            ]
        
        )
        
        print(f"Collection '{collection_name}' created successfully with vectorizer.")
    
    def insert_data(self, client, collection_name:str, data:Dict):
        # Insert data into the specified collection
        collection = client.collections.get(collection_name)
        
        for disease, meta in data.items():
            collection.data.insert({
                "disease_name": disease,
                "possible_condition": meta["possible condition"],
                "severity_level": meta["Severity level"],
                "first_aid": meta["First aid Instruction"],
                "otc_recommendations": meta["OTC product recommendations"],
                "monitoring": meta["Monitoring"],
                "vet_visit_urgency": meta["vet visit urgency"],
                "emergency_flag": meta["emergency flag"] == "True"
            })
        
        print("All diseases inserted successfully")

        print(f"Data inserted into collection '{collection_name}': {data}")