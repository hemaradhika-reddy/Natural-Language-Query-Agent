import json
from langchain import Document
from sentence_transformers import SentenceTransformer

def prepare_data(lecture_notes_path, architectures_path):
    # Load and process lecture notes
    with open(lecture_notes_path, 'r') as file:
        lecture_notes = file.read()
    
    # Load and process LLM architectures
    with open(architectures_path, 'r') as file:
        architectures = json.load(file)
    
    documents = [Document(content=lecture_notes)]
    for arch in architectures:
        documents.append(Document(content=json.dumps(arch)))
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(doc.content) for doc in documents]
    
    return documents, embeddings
