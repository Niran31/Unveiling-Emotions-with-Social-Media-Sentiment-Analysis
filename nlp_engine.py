from transformers import pipeline
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import re

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
