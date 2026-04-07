import openai
import json
import faiss
import numpy as np
import gradio as gr
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sentence_transformers import CrossEncoder  # SBERT Cross-Encoder

# In order to clean our entries we need to download this frist
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load SBERT
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Set up OpenAI API key
openai.api_key = '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'  # ✅ REMOVE your actual API key for security

# Load merged library data
with open('json/library_data.json', 'r', encoding='utf-8') as f:
    library_data = json.load(f)

# Extract questions, answers, and book details
entries = [entry for entry in library_data]
texts = [entry['text'] for entry in entries]

# Input queries cleaning function
def clean_text(text):
    """Preprocess text by lowercasing, removing punctuation, and filtering stop-words."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop-words
    return text

cleaned_texts = [clean_text(text) for text in texts]

# Metadata mapping, useful to identify titles in our FAQ
metadata_mapping = {i: entry for i, entry in enumerate(library_data)}

# Create embeddings
def generate_embeddings(texts):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [embedding['embedding'] for embedding in response['data']]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.zeros((len(texts), 1536)).tolist()

combined_embeddings = generate_embeddings(cleaned_texts)
normalized_combined_embeddings = normalize(combined_embeddings)
embedding_array = np.array(normalized_combined_embeddings, dtype=np.float32)

# Create FAISS index
dimension = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embedding_array)

# Function to retrieve and rerank information
def retrieve_information(query):
    try:
        cleaned_query = clean_text(query)  # Text cleaned
        query_embedding = generate_embeddings([cleaned_query])
        query_embedding = normalize(query_embedding)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        _, indices = faiss_index.search(query_embedding, k=10) # Retrieve top 10 closest matches from FAISS
        retrieved_entries = [metadata_mapping[idx] for idx in indices[0]]

        if not retrieved_entries:
            return "❌ Sorry, I couldn't find relevant information."

        # Reranking
        rerank_inputs = [(cleaned_query, entry['text'] + (" FAQ" if entry['type'] == 'faq' else "")) 
                         for entry in retrieved_entries
]

        scores = reranker.predict(rerank_inputs)        
        best_match_idx = scores.argmax() # Best match (highest score)
        best_entry = retrieved_entries[best_match_idx]

        # Return the most relevant text
        if best_entry['type'] == 'faq':
            return best_entry['answer']
        elif best_entry['type'] == 'book':
            return f"📖 {best_entry['title']}: {best_entry['status']}"
    
    except Exception as e:
        print(f"Error retrieving information: {e}")
        return "❌ I encountered an issue while searching for information."

# Generate response using GPT-3.5
def get_gpt_response(query, chat_history):
    cleaned_query = clean_text(query)
    retrieved_info = retrieve_information(cleaned_query)
    formatted_chat_history = [{"role": "assistant", "content": entry} for entry in chat_history if isinstance(entry, str)]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Librito, an AI assistant for the Vistula University Library. "
                    "Always use the retrieved information before generating a response. If you cannot find an answer, say 'I don't know'."},
                *formatted_chat_history,
                {"role": "user", "content": f"User's question: {cleaned_query}. Relevant library information: {retrieved_info}"},
            ],
            max_tokens=200,
            temperature=0.5
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "❌ Apologies, I encountered an issue generating a response."

# Gradio Chat Interface
chatbot = gr.ChatInterface(fn=get_gpt_response, title="Vistula University Library Chatbot")

# Launch Chat Interface
if __name__ == "__main__":
    chatbot.launch(share=True)