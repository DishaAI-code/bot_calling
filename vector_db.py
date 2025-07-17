from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import os

class VectorDB:
    def __init__(self, db_path="chroma_db"):
        self.client = PersistentClient(path=db_path)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name="lpu_courses",
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}  # Better for semantic search
        )
    
    def has_courses(self):
        """Check if collection already contains courses"""
        return len(self.collection.get()['ids']) > 0
    
    def add_courses(self, new_courses):
        """Only add courses that don't already exist"""
        existing = set(self.collection.get()['documents'])
        to_add = [c for c in new_courses if c not in existing]
        
        if to_add:
            new_ids = [str(len(existing) + i) for i in range(len(to_add))]
            self.collection.add(
                ids=new_ids,
                documents=to_add
            )
            print(f"Added {len(to_add)} new courses to DB")
        else:
            print("All courses already exist in DB")
    
    def search_courses(self, query, n_results=5):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0]
        except Exception as e:
            print(f"Search error: {e}")
            return []