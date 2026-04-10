from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nlp_engine import get_sentiment, analyze_batch, mine_frequent_patterns, extract_entities, extract_keywords
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
    sentiment = get_sentiment(text)
    return jsonify({'text': text, "sentiment": sentiment})

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch_endpoint():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                text_col = next((col for col in df.columns if col.lower() == 'text'), None)
                if not text_col:
                    return jsonify({'error': 'CSV must contain a text column'}), 400
                
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

                # NEW: Save to History
                top_emotion = max(analysis['counts'], key=analysis['counts'].get) if analysis['counts'] else 'N/A'
                save_to_history('CSV Upload', file.filename, len(texts), analysis['counts'], top_emotion, entities[:5])
                    
                return jsonify(analysis)
            except Exception as e:
                return jsonify({'error': f'Failed processing file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/api/scrape', methods=['GET'])
def scrape_live():
    topic = request.args.get('topic', 'technology')
    safe_topic = urllib.parse.quote(topic)
    url = f"https://www.reddit.com/search.rss?q={safe_topic}&sort=new"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Tweetverse/2.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': f'Failed to scrape live data. Status {response.status_code}'}), 500
            
        root = ET.fromstring(response.content)
        texts = []
        # Parse Atom feed XML
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:25]:
            title_node = entry.find("{http://www.w3.org/2005/Atom}title")
            if title_node is not None and title_node.text:
                texts.append(title_node.text)
            
        if not texts:
            return jsonify({'error': 'No internet buzz found for this topic.'}), 404
            
        analysis = analyze_batch(texts)
        for i, res in enumerate(analysis['results']):
            res['id'] = f"live-{i+1}"
            
        # Data Mining
        patterns = mine_frequent_patterns(texts)
        analysis['apriori_rules'] = patterns

        # NEW: Named Entity Recognition
        entities = extract_entities(texts)
        analysis['entities'] = entities

        # NEW: Keyword Extraction for Word Cloud
        keywords = extract_keywords(texts)
        analysis['keywords'] = keywords
        
        # NEW: Save to History
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
