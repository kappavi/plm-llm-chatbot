import faiss
import numpy as np
from typing import Dict, List, Optional
import json
import os

class VectorRetriever:
    def __init__(self, index_path: Optional[str] = None):
        self.dimension = 1280
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def add_documents(self, documents: List[Dict]):
        embeddings = np.array([doc['embeddings'] for doc in documents])
        self.index.add(embeddings)
        self.documents.extend(documents)

    def retrieve(self, query: str, context: Dict, k: int = 5) -> Dict:
        query_embedding = context['embeddings']

        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            k
        )

        relevant_docs = [self.documents[i] for i in indices[0]]

        return {
            'relevant_documents': relevant_docs,
            'distances': distances[0].tolist()
        }

    def save_index(self, path: str):
        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/documents.json", 'w') as f:
            json.dump(self.documents, f)

    def load_index(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/documents.json", 'r') as f:
            self.documents = json.load(f)