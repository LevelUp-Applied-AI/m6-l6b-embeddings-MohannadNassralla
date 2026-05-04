import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

def build_tfidf(texts):
    """
    Build TF-IDF representations for a list of texts.
    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """
    Compute pairwise cosine similarity from a TF-IDF matrix.
    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """
    Load pre-trained GloVe vectors from a text file.
    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
    except FileNotFoundError:
        return None
    return embeddings


def text_to_glove(text, embeddings):
    """
    Compute the average GloVe embedding for a text.
    Skip out-of-vocabulary words. Returns a vector of shape (50,).
    """
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    
    if not vectors:
        return np.zeros(50,)
    
    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """
    Extract a sentence embedding from DistilBERT using mean pooling.
    Returns a numpy array of shape (768,).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # last_hidden_state has shape [batch_size, seq_len, hidden_size]
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    # Expand mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum embeddings while ignoring padding, then divide by the sum of the mask
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    embedding = (sum_embeddings / sum_mask).cpu().numpy().flatten()
    return embedding


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings, 
                        bert_model, bert_tokenizer):
    """
    Compare similarity rankings across TF-IDF, GloVe, and BERT.
    """
    results = {}
    
    # Pre-compute GloVe and BERT embeddings for the entire corpus for efficiency
    glove_corpus = np.array([text_to_glove(t, glove_embeddings) for t in texts])
    bert_corpus = np.array([extract_bert_embedding(t, bert_tokenizer, bert_model) for t in texts])
    
    for query in queries:
        # 1. Get TF-IDF scores
        try:
            q_idx = texts.index(query)
            tfidf_scores = tfidf_sim[q_idx]
        except ValueError:
            tfidf_scores = np.zeros(len(texts))

        # 2. Compute GloVe similarity
        q_glove = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, glove_corpus).flatten()
        
        # 3. Compute BERT similarity
        q_bert = extract_bert_embedding(query, bert_tokenizer, bert_model).reshape(1, -1)
        bert_scores = sklearn_cosine(q_bert, bert_corpus).flatten()
        
        methods = {
            "tfidf": tfidf_scores,
            "glove": glove_scores,
            "bert": bert_scores
        }
        
        query_results = {}
        for method_name, scores in methods.items():
            # Get top indices, excluding the query itself (similarity of 1.0)
            sorted_indices = np.argsort(scores)[::-1]
            top_3 = []
            for idx in sorted_indices:
                if texts[idx] != query:
                    top_3.append((texts[idx], float(scores[idx])))
                if len(top_3) == 3:
                    break
            query_results[method_name] = top_3
            
        results[query] = query_results
        
    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load data
    try:
        df = pd.read_csv("data/bbc_news.csv")
        texts = df["text"].tolist()
        print(f"Loaded {len(texts)} texts")
    except FileNotFoundError:
        print("Data file not found.")
        texts = []

    if texts:
        # Task 1: TF-IDF
        result = build_tfidf(texts)
        if result:
            tfidf_matrix, vectorizer = result
            print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            tfidf_sim = compute_tfidf_similarity(tfidf_matrix)

        # Task 2: GloVe
        glove = load_glove("data/glove_50k_50d.txt")
        if glove:
            print(f"Loaded {len(glove)} GloVe vectors")

        # Task 3: DistilBERT
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()

        # Task 4: Compare
        if result and glove:
            # Pick one query per category
            queries = [df[df["category"] == cat]["text"].iloc[0]
                       for cat in df["category"].unique()]
            
            comparison = compare_similarities(
                texts, queries, tfidf_sim, glove, model, tokenizer
            )
            
            if comparison:
                for q in list(comparison.keys()):
                    print(f"\nQuery: {q[:80]}...")
                    for method in ["tfidf", "glove", "bert"]:
                        top = comparison[q].get(method, [])
                        print(f"  {method}: {[t[:40].replace('\\n', ' ') for t, _ in top]}")