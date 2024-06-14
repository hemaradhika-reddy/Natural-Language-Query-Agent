from transformers import pipeline
from langchain import LangChain

def handle_query(query, documents, embeddings):
    # Load pre-trained model and tokenizer
    nlp = pipeline('question-answering')
    
    # Use LangChain to retrieve relevant documents
    langchain = LangChain()
    relevant_docs = langchain.retrieve_documents(query, documents, embeddings)
    
    # Generate answer using the pre-trained model
    answers = [nlp({'question': query, 'context': doc.content}) for doc in relevant_docs]
    return answers
