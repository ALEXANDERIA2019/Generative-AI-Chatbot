import openai
import json
import faiss
import numpy as np
import gradio as gr
from sklearn.preprocessing import normalize

# Set up OpenAI API key
openai.api_key = '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'

# Load merged library data
with open('json/library_data.json', 'r', encoding='utf-8') as f:
    library_data = json.load(f)

# Extract questions, answers, and book details
entries = [entry for entry in library_data]
texts = [entry['text'] for entry in entries]

# Create a metadata mapping
metadata_mapping = {i: entry for i, entry in enumerate(entries)}

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
combined_embeddings = generate_embeddings(texts)
normalized_combined_embeddings = normalize(combined_embeddings)
embedding_array = np.array(normalized_combined_embeddings, dtype=np.float32)

# Create FAISS index
dimension = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embedding_array)

# Function to retrieve information
def retrieve_information(query):
    try:
        query_embedding = generate_embeddings([query])
        query_embedding = normalize(query_embedding)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        _, indices = faiss_index.search(query_embedding, k=1)
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
    retrieved_info = retrieve_information(query)
    formatted_chat_history = [{"role": "assistant", "content": entry} for entry in chat_history if isinstance(entry, str)]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Librito, an AI assistant for the Vistula University Library. "
                 "Answer naturally, using the retrieved information from the library's FAQ and book catalog."},
                *formatted_chat_history,
                {"role": "user", "content": query},
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
chatbot = gr.ChatInterface(
    fn=get_gpt_response,
    title="Vistula University Library Chatbot",
    description="Hi! I'm Librito, AI chatbot of Vistula University's Library. I'll help with book searches, research help, library information, and more!",
    theme="default",
    chatbot=gr.Chatbot(label="Chat History", type="messages"),
    textbox=gr.Textbox(placeholder="Type your question here..."),
    submit_btn="Ask",
    stop_btn="Stop"
)

# Launch Chat Interface
if __name__ == "__main__":
    chatbot.launch(share=True)