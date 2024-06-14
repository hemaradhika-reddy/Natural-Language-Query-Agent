from data_preparation import prepare_data
from query_handler import handle_query

# Prepare data
documents, embeddings = prepare_data('data/lecture_notes.txt', 'data/llm_architectures.json')

# Handle a sample query
query = "What are some milestone model architectures and papers in the last few years?"
answers = handle_query(query, documents, embeddings)

# Print answers
for answer in answers:
    print(answer)
