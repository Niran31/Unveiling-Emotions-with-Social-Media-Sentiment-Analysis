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
        return {"label": "Neutral", "confidence": 1.0}
    
    try:
        # Transformers have token limits. Truncate text.
        results = emotion_classifier(text[:512]) 
        pred = results[0][0]
        label = pred['label'].capitalize()
        return {"label": label, "confidence": float(pred['score'])}
    except Exception as e:
        print(f"DL Error: {e}")
        return {"label": "Neutral", "confidence": 1.0}

def analyze_batch(texts):
    results = []
    counts = {}
    confidences = []
    
    for text in texts:
        res = get_sentiment(text)
        label = res['label']
        conf = res['confidence']
        
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
        results.append({"text": text, "sentiment": label, "confidence": conf})
        confidences.append(conf)
    
    # Ensure some standard colors exist even if 0
    for basic in ['Joy', 'Anger', 'Sadness', 'Fear', 'Surprise', 'Love', 'Neutral']:
        if basic not in counts and len(counts) < 3: # Keep clean if none found
             pass
             
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {"results": results, "counts": counts, "avg_confidence": avg_conf}

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

        # Apriori: Increase min_support and limit max_len to prevent exponential CPU freeze on dense text
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True, max_len=3)
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


# =============================================
# NEW: Aspect-Based Sentiment Analysis (ABSA)
# =============================================
def extract_aspects(texts, top_n=12):
    """
    Extracts noun-phrase aspects from texts using SpaCy chunking,
    then classifies the sentiment of each aspect's surrounding sentence.
    Returns a list of top aspects with per-emotion breakdowns.
    """
    aspect_counter = Counter()
    aspect_sentiments = {}  # aspect_key -> Counter of emotions

    stop_aspects = {
        'it', 'i', 'we', 'they', 'you', 'he', 'she', 'this', 'that',
        'which', 'what', 'who', 'there', 'here', 'these', 'those',
        'me', 'us', 'them', 'my', 'our', 'your', 'its', 'the',
        'a', 'an', 'some', 'any', 'all', 'no', 'every', 'each'
    }

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        doc = nlp_spacy(text[:1000])

        # Extract noun chunks as aspects
        seen_in_text = set()
        for chunk in doc.noun_chunks:
            # Clean and normalize
            aspect = chunk.text.strip().lower()
            # Remove leading determiners/articles
            aspect = re.sub(r'^(the|a|an|this|that|my|our|your|his|her|its|their)\s+', '', aspect)
            aspect = aspect.strip()

            if len(aspect) < 2 or aspect in stop_aspects or aspect.isdigit():
                continue
            if aspect in seen_in_text:
                continue
            seen_in_text.add(aspect)

            aspect_counter[aspect] += 1

            # Get the sentiment of the sentence containing this chunk
            sent_text = chunk.sent.text if chunk.sent else text
            result = get_sentiment(sent_text[:512])
            label = result['label']

            if aspect not in aspect_sentiments:
                aspect_sentiments[aspect] = Counter()
            aspect_sentiments[aspect][label] += 1

    # Build output for the top N most-mentioned aspects
    output = []
    for aspect, count in aspect_counter.most_common(top_n):
        sentiments = dict(aspect_sentiments.get(aspect, {}))
        dominant = max(sentiments, key=sentiments.get) if sentiments else 'Neutral'
        output.append({
            "aspect": aspect,
            "count": count,
            "sentiments": sentiments,
            "dominant": dominant
        })

    return output
