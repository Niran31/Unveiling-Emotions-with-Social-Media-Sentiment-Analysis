from transformers import pipeline
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import re
from collections import Counter

# --- SpaCy NER Setup ---
import spacy
try:
    nlp_spacy = spacy.load("en_core_web_sm")
    print("[NER] SpaCy en_core_web_sm model loaded.")
except OSError:
    print("[NER] SpaCy model not found. Downloading en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")
    print("[NER] SpaCy en_core_web_sm model loaded after download.")

print("--------------------------------------------------")
print("Loading Deep Learning Model (First run may take a minute to download)...")
# Load a multi-class emotion model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
print("Model Loaded Successfully!")
print("--------------------------------------------------")

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
    
    try:
        # Transformers have token limits. Truncate text.
        results = emotion_classifier(text[:512]) 
        pred = results[0][0]
        label = pred['label'].capitalize()
        return label
    except Exception as e:
        print(f"DL Error: {e}")
        return "Neutral"

def analyze_batch(texts):
    results = []
    counts = {}
    for text in texts:
        sentiment = get_sentiment(text)
        if sentiment not in counts:
            counts[sentiment] = 0
        counts[sentiment] += 1
        results.append({"text": text, "sentiment": sentiment})
    
    # Ensure some standard colors exist even if 0
    for basic in ['Joy', 'Anger', 'Sadness', 'Fear', 'Surprise', 'Love', 'Neutral']:
        if basic not in counts and len(counts) < 3: # Keep clean if none found
             pass
             
    return {"results": results, "counts": counts}

def mine_frequent_patterns(texts):
    """ Executes the Apriori algorithm on texts """
    if not texts or len(texts) < 2:
        return []

    dataset = []
    stop_words = set(['the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'it', 'that', 'with', 'on', 'this', 'i', 'my', 'you', 'are', 'be', 'was', 'as'])
    
    for text in texts:
        if not isinstance(text, str): continue
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        meaningful_words = [w for w in words if w not in stop_words]
        if meaningful_words:
            dataset.append(meaningful_words)
            
    if not dataset:
        return []

    try:
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Apriori: min_support 0.05 because texts are sparse
        frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
        if len(frequent_itemsets) == 0:
            return []
            
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
        rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        output_rules = []
        for _, row in rules.head(10).iterrows():
            output_rules.append({
                "antecedents": list(row['antecedents']),
                "consequents": list(row['consequents']),
                "support": round(row['support'], 3),
                "confidence": round(row['confidence'], 3),
                "lift": round(row['lift'], 3)
            })
        return output_rules
    except Exception as e:
        print(f"Apriori Error: {e}")
        return []


# =============================================
# NEW: Named Entity Recognition (NER)
# =============================================
def extract_entities(texts):
    """
    Uses SpaCy NER to extract and rank named entities from a list of texts.
    Returns the top 15 entities with their type and frequency count.
    """
    entity_counter = Counter()
    entity_labels = {}  # Store the NER label for each entity

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        doc = nlp_spacy(text[:1000])  # Truncate for performance
        for ent in doc.ents:
            # Filter out very short or numeric-only entities
            clean = ent.text.strip()
            if len(clean) < 2 or clean.isdigit():
                continue
            # Normalize casing for better grouping
            key = clean.title()
            entity_counter[key] += 1
            entity_labels[key] = ent.label_

    # Return the top 15 most mentioned entities
    top_entities = []
    for name, count in entity_counter.most_common(15):
        top_entities.append({
            "name": name,
            "type": entity_labels.get(name, "MISC"),
            "count": count
        })
    return top_entities


# =============================================
# NEW: Keyword Frequency for Word Cloud
# =============================================
def extract_keywords(texts, top_n=60):
    """
    Extracts the most frequent meaningful keywords from a list of texts.
    Returns a list of {word, count} for the Word Cloud visualization.
    """
    stop_words = set([
        'the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'it', 'that', 'with',
        'on', 'this', 'i', 'my', 'you', 'are', 'be', 'was', 'as', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
        'may', 'might', 'shall', 'not', 'but', 'or', 'if', 'then', 'than', 'too',
        'very', 'just', 'about', 'up', 'out', 'so', 'no', 'all', 'some', 'any',
        'each', 'every', 'only', 'own', 'more', 'other', 'into', 'over', 'such',
        'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why', 'been',
        'being', 'here', 'there', 'they', 'them', 'their', 'his', 'her', 'its',
        'our', 'your', 'we', 'he', 'she', 'me', 'him', 'us', 'from', 'at', 'by',
        'an', 'were', 'also', 'like', 'get', 'got', 'don', 'amp', 'one', 'two',
        'even', 'still', 'way', 'much', 'going', 'really', 'right', 'back', 'now',
        'well', 'off', 'let', 'say', 'said', 'new', 'see', 'want', 'come', 'make',
        'think', 'know', 'take', 'go', 'thing', 'things', 'https', 'http', 'www',
        'com', 'use', 'used'
    ])

    word_counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        meaningful = [w for w in words if w not in stop_words]
        word_counter.update(meaningful)

    results = []
    for word, count in word_counter.most_common(top_n):
        results.append({"word": word, "count": count})
    return results
