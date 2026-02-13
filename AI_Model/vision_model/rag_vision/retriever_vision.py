from dotenv import load_dotenv
import os
load_dotenv()

from pinecone import Pinecone
from openai import OpenAI
from typing import Dict, List, Union

def decode_results(results) -> Union[Dict, List[Dict]]:
    """
    Decode Pinecone FetchResponse or QueryResponse into clean dictionaries.
    
    Args:
        results: FetchResponse or QueryResponse from Pinecone
        
    Returns:
        dict or list of dicts containing disease information
    """
    # Handle FetchResponse (from index.fetch())
    if hasattr(results, 'vectors') and isinstance(results.vectors, dict):
        decoded = []
        for vector_id, vector in results.vectors.items():
            if vector.metadata:
                decoded.append({
                    'id': vector_id,
                    'metadata': vector.metadata
                })
        # Return single dict if only one result, else list
        return decoded[0] if len(decoded) == 1 else decoded
    
    # Handle QueryResponse (from index.query())
    elif hasattr(results, 'matches') and isinstance(results.matches, list):
        decoded = []
        for match in results.matches:
            decoded.append({
                'id': match.get('id'),
                'score': match.get('score'),
                'metadata': match.get('metadata', {})
            })
        return decoded
    
    # Fallback
    return results


def retrieve_docs(query: str, index_or_hostname):
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    
    # Check if it's a hostname (URL) or index name
    if index_or_hostname.startswith("http"):
        index = pc.Index(host=index_or_hostname)
    else:
        index = pc.Index(name=index_or_hostname)
    
    # Try exact ID match with various formats
    possible_ids = [
        query.lower(),
        query.replace(" ", "_").lower(), 
        query.replace(" ", "-").lower(),
        query.title().lower(),
        query
    ]
    
    try:
        results = index.fetch(
            ids=possible_ids,
            namespace="__default__"
        )
        
        # If exact ID match found, return it
        if results and results.vectors:
            return decode_results(results)
    except Exception as e:
        print(f"ID fetch failed: {e}")
    
    # Fall back to vector similarity search
    client = OpenAI(api_key=os.getenv("OPENAIAPI"))
    embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-large",
        dimensions=512
    )
    
    results = index.query(
        namespace="__default__",
        vector=embedding.data[0].embedding,
        top_k=5,
        include_metadata=True
    )
    
    return decode_results(results)


if __name__ == "__main__":
    # Example usage
    query = "fur loss cushing disease"
    index_name = "dog-disease"
    
    # Get raw results
    raw_results = retrieve_docs(query, index_name)
    
    # Decode into clean format
    decoded_results = decode_results(raw_results)
    
    # Pretty print
    import json
    print(json.dumps(decoded_results, indent=2))