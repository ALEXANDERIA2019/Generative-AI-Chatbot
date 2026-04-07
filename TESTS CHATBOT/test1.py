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

# In order to clean our entries we need to download this frist
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set up OpenAI API key
openai.api_key = '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'

# Load merged library data
with open('json/library_data.json', 'r', encoding='utf-8') as f:
    library_data = json.load(f)

# Extract questions, answers, and book details
entries = [entry for entry in library_data]
texts = [entry['text'] for entry in entries]

# Cleaning function
def clean_text(text):
    """Preprocess text by lowercasing, removing punctuation, and filtering stop-words."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop-words
    return text

# Clean dataset before storing embeddings
cleaned_texts = [clean_text(text) for text in texts]

# Create a metadata mapping
metadata_mapping = {i: entry for i, entry in enumerate(library_data)}

# Function to generate embeddings
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

# Generate embeddings
combined_embeddings = generate_embeddings(cleaned_texts)
normalized_combined_embeddings = normalize(combined_embeddings)
embedding_array = np.array(normalized_combined_embeddings, dtype=np.float32)

# Create FAISS index
dimension = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embedding_array)

# Function to retrieve information
def retrieve_information(query):
    try:
        cleaned_query = clean_text(query)  # Query cleaned before searching FAISS
        query_embedding = generate_embeddings([cleaned_query])
        query_embedding = normalize(query_embedding)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        _, indices = faiss_index.search(query_embedding, k=3) # K modify how many closests matches we want
        best_match_idx = indices[0][0]

        item = metadata_mapping.get(best_match_idx, None)
        if not item:
            return "❌ Sorry, I couldn't find relevant information."

        if item['type'] == 'faq':
            return item['answer']
        elif item['type'] == 'book':
            return f"📖 {item['title']}: {item['status']}"
    except Exception as e:
        print(f"Error retrieving information: {e}")
        return "❌ I encountered an issue while searching for information."

# Function to generate response using GPT-3.5
def get_gpt_response(query, chat_history):
    cleaned_query = clean_text(query)
    retrieved_info = retrieve_information(cleaned_query)
    formatted_chat_history = [{"role": "assistant", "content": entry} for entry in chat_history if isinstance(entry, str)]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Librito, an AI assistant for the Vistula University Library."},
                *formatted_chat_history,
                {"role": "user", "content": cleaned_query},
                {"role": "assistant", "content": retrieved_info}
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