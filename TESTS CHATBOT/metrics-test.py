import time
import json
import numpy as np
import faiss
import openai
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sentence_transformers import CrossEncoder
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
from sklearn.metrics import ndcg_score

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load test dataset
with open('test_queries.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)  # Contains [{"query": "...", "expected_answer": "..."}, ...]

# Load library data
with open('json/library_data.json', 'r', encoding='utf-8') as f:
    library_data = json.load(f)

# Extract metadata
metadata_mapping = {i: entry for i, entry in enumerate(library_data)}
texts = [entry['text'] for entry in library_data]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# OpenAI API Key (ensure security)
openai.api_key = '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'

def generate_embeddings(texts):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
    return [embedding['embedding'] for embedding in response['data']]

# Create embeddings and FAISS index
cleaned_texts = [clean_text(text) for text in texts]
combined_embeddings = generate_embeddings(cleaned_texts)
normalized_embeddings = normalize(combined_embeddings)
embedding_array = np.array(normalized_embeddings, dtype=np.float32)

# Build FAISS index
dimension = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embedding_array)

def retrieve_top_k(query, k=5):
    cleaned_query = clean_text(query)
    query_embedding = generate_embeddings([cleaned_query])
    query_embedding = normalize(query_embedding)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    _, indices = faiss_index.search(query_embedding, k)
    retrieved_entries = [metadata_mapping[idx] for idx in indices[0]]

    # Rerank using SBERT CrossEncoder
    rerank_inputs = [(cleaned_query, entry['text']) for entry in retrieved_entries]
    scores = reranker.predict(rerank_inputs)
    reranked_entries = sorted(zip(retrieved_entries, scores), key=lambda x: x[1], reverse=True)
    return [entry[0] for entry in reranked_entries]

def evaluate_retrieval():
    precision_list, recall_list, mrr_list, ndcg_list = [], [], [], []
    
    for data in test_data:
        query = data['query']
        expected_answer = data['expected_answer'].lower()
        retrieved_entries = retrieve_top_k(query, k=10)

        retrieved_texts = [entry['text'].lower() for entry in retrieved_entries]

        # Create relevance scores: 1 if text is relevant, 0 otherwise
        relevance_scores = [1 if expected_answer in text else 0 for text in retrieved_texts]

        relevant_count = sum(relevance_scores)

        # Compute metrics
        precision_at_5 = relevant_count / 5.0
        recall_at_10 = relevant_count / len(retrieved_entries)
        mrr = 1.0 / (relevance_scores.index(1) + 1) if 1 in relevance_scores else 0

        # Fix ndcg_score by passing numerical relevance values
        ndcg = ndcg_score([relevance_scores], [sorted(relevance_scores, reverse=True)])

        precision_list.append(precision_at_5)
        recall_list.append(recall_at_10)
        mrr_list.append(mrr)
        ndcg_list.append(ndcg)

    print(f"Precision@5: {np.mean(precision_list):.4f}")
    print(f"Recall@10: {np.mean(recall_list):.4f}")
    print(f"MRR: {np.mean(mrr_list):.4f}")
    print(f"nDCG: {np.mean(ndcg_list):.4f}")

def evaluate_generation():
    rouge, bleu, bert = [], [], []
    
    for data in test_data:
        query = data['query']
        expected_answer = data['expected_answer']
        generated_answer = get_gpt_response(query, [])

        # Compute ROUGE
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score_value = scorer.score(expected_answer, generated_answer)['rougeL'].fmeasure
        
        # Compute BLEU
        bleu_score_value = sentence_bleu([expected_answer.split()], generated_answer.split())
        
        # Compute BERTScore
        P, R, F1 = bert_score([generated_answer], [expected_answer], lang="en")
        bert_score_value = F1.mean().item()

        rouge.append(rouge_score_value)
        bleu.append(bleu_score_value)
        bert.append(bert_score_value)

    print(f"ROUGE-L: {np.mean(rouge):.4f}")
    print(f"BLEU: {np.mean(bleu):.4f}")
    print(f"BERTScore: {np.mean(bert):.4f}")

def get_gpt_response(query, chat_history):
    messages = [
        {"role": "system", "content": "You are Librito, the Vistula University Library assistant."},
        {"role": "user", "content": query}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    print("Evaluating Retrieval...")
    evaluate_retrieval()
    print("\nEvaluating Generation...")
    evaluate_generation()