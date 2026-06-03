from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nlp_engine import get_sentiment, analyze_batch, mine_frequent_patterns, extract_entities, extract_keywords, extract_aspects
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import urllib.parse
import sqlite3
import json
import os
from datetime import datetime

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# =============================================
# SQLite Database Setup
# =============================================
DB_PATH = os.path.join(os.path.dirname(__file__), 'tweetverse_history.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            topic TEXT,
            total_items INTEGER,
            counts_json TEXT,
            top_emotion TEXT,
            entities_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_to_history(source, topic, total_items, counts, top_emotion, entities):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO analysis_history (timestamp, source, topic, total_items, counts_json, top_emotion, entities_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            source,
            topic or '',
            total_items,
            json.dumps(counts),
            top_emotion,
            json.dumps(entities)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Write Error: {e}")


@app.route('/')
def index():
    return send_from_directory('.', 'sentiment_app.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_single():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    sentiment_res = get_sentiment(text)
    return jsonify({
        'text': text, 
        "sentiment": sentiment_res['label'],
        "confidence": sentiment_res['confidence']
    })

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch_endpoint():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                
                # Flexible text column detection
                possible_cols = ['text', 'tweet', 'message', 'comment', 'body', 'post', 'content', 'sentence', 'review']
                text_col = next((col for col in df.columns if col.lower() in possible_cols), None)
                if not text_col:
                    # Fallback: locate first object/string type column that is not 'id'
                    for col in df.columns:
                        if col.lower() != 'id' and df[col].dtype == 'object':
                            text_col = col
                            break
                            
                if not text_col:
                    return jsonify({'error': 'Could not identify a valid text column in the CSV file.'}), 400
                
                texts = df[text_col].tolist()
                id_col = next((col for col in df.columns if col.lower() == 'id'), None)
                ids = df[id_col].tolist() if id_col else list(range(1, len(texts) + 1))
                
                analysis = analyze_batch(texts)
                for i, res in enumerate(analysis['results']):
                    res['id'] = ids[i]
                
                # Perform Data Mining Apriori
                patterns = mine_frequent_patterns(texts)
                analysis['apriori_rules'] = patterns

                # NEW: Named Entity Recognition
                entities = extract_entities(texts)
                analysis['entities'] = entities

                # NEW: Keyword Extraction for Word Cloud
                keywords = extract_keywords(texts)
                analysis['keywords'] = keywords

                # NEW: Aspect-Based Sentiment Analysis
                aspects = extract_aspects(texts)
                analysis['aspects'] = aspects

                # NEW: Save to History
                top_emotion = max(analysis['counts'], key=analysis['counts'].get) if analysis['counts'] else 'N/A'
                save_to_history('CSV Upload', file.filename, len(texts), analysis['counts'], top_emotion, entities[:5])
                    
                return jsonify(analysis)
            except Exception as e:
                return jsonify({'error': f'Failed processing file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid request'}), 400

# =============================================
# Multi-Source Scraper Helpers
# =============================================
import re as re_module

def scrape_wikipedia(topic):
    """Scrape Wikipedia search results for a topic."""
    safe_topic = urllib.parse.quote(topic)
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={safe_topic}&utf8=&format=json&srlimit=15"
    try:
        headers = {'User-Agent': 'Tweetverse/3.0 (student@university.edu)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        texts = []
        for item in data.get('query', {}).get('search', []):
            snippet = item.get('snippet', '')
            clean_text = re_module.sub(r'<[^>]+>', '', snippet)
            if clean_text:
                texts.append({'text': f"{item.get('title')}: {clean_text}", 'source': 'Wikipedia'})
        return texts
    except Exception as e:
        print(f"[Wikipedia] Scrape error: {e}")
        return []

def scrape_reddit(topic):
    """Scrape Reddit search results for a topic via public JSON API."""
    try:
        url = f"https://www.reddit.com/search.json?q={urllib.parse.quote(topic)}&limit=15&sort=relevance&t=month"
        headers = {'User-Agent': 'Tweetverse/3.0 Sentiment Bot'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        texts = []
        for child in data.get('data', {}).get('children', []):
            post = child.get('data', {})
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            combined = f"{title}. {selftext}" if selftext else title
            combined = combined.strip()
            if combined and len(combined) > 10:
                texts.append({'text': combined[:500], 'source': 'Reddit'})
        return texts
    except Exception as e:
        print(f"[Reddit] Scrape error: {e}")
        return []

def scrape_hackernews(topic):
    """Scrape HackerNews via Algolia public API."""
    try:
        url = f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(topic)}&hitsPerPage=10&tags=story"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        data = response.json()
        texts = []
        for hit in data.get('hits', []):
            title = hit.get('title', '')
            # Some stories have story_text (for Ask HN etc.)
            story_text = hit.get('story_text', '')
            if story_text:
                clean = re_module.sub(r'<[^>]+>', '', story_text)
                combined = f"{title}. {clean}"
            else:
                combined = title
            combined = combined.strip()
            if combined and len(combined) > 5:
                texts.append({'text': combined[:500], 'source': 'HackerNews'})
        return texts
    except Exception as e:
        print(f"[HackerNews] Scrape error: {e}")
        return []


@app.route('/api/scrape', methods=['GET'])
def scrape_live():
    topic = request.args.get('topic', 'technology')
    
    try:
        # Aggregate from multiple sources
        raw_results = []
        raw_results += scrape_wikipedia(topic)
        raw_results += scrape_reddit(topic)
        raw_results += scrape_hackernews(topic)

        if not raw_results:
            return jsonify({'error': 'No internet buzz found for this topic across any source.'}), 404

        texts = [r['text'] for r in raw_results]
        sources = [r['source'] for r in raw_results]

        # Compute source distribution counts
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1
            
        analysis = analyze_batch(texts)
        for i, res in enumerate(analysis['results']):
            res['id'] = f"live-{i+1}"
            res['source'] = sources[i]
            
        analysis['source_counts'] = source_counts

        # Data Mining
        patterns = mine_frequent_patterns(texts)
        analysis['apriori_rules'] = patterns

        # Named Entity Recognition
        entities = extract_entities(texts)
        analysis['entities'] = entities

        # Keyword Extraction for Word Cloud
        keywords = extract_keywords(texts)
        analysis['keywords'] = keywords

        # Aspect-Based Sentiment Analysis
        aspects = extract_aspects(texts)
        analysis['aspects'] = aspects
        
        # Save to History
        top_emotion = max(analysis['counts'], key=analysis['counts'].get) if analysis['counts'] else 'N/A'
        save_to_history('Live Scan', topic, len(texts), analysis['counts'], top_emotion, entities[:5])

        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': f"Scraping error: {str(e)}"}), 500


# =============================================
# NEW: History API
# =============================================
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM analysis_history ORDER BY id DESC LIMIT 20')
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'source': row['source'],
                'topic': row['topic'],
                'total_items': row['total_items'],
                'counts': json.loads(row['counts_json']),
                'top_emotion': row['top_emotion'],
                'entities': json.loads(row['entities_json'])
            })
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.cursor().execute('DELETE FROM analysis_history')
        conn.commit()
        conn.close()
        return jsonify({'status': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
