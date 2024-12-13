from flask import Flask, request, jsonify, render_template_string
import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

HACKER_NEWS_API = "https://hacker-news.firebaseio.com/v0/"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_top_stories():
    response = requests.get(f"{HACKER_NEWS_API}topstories.json")
    return response.json()[:500]

def get_story_details(story_id):
    response = requests.get(f"{HACKER_NEWS_API}item/{story_id}.json")
    return response.json()

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing special characters, and extra spaces.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def rank_stories(user_bio, stories):
    """
    Rank stories based on their similarity to the user bio using sentence embeddings.
    """
    # Preprocess the user bio
    user_bio_processed = preprocess_text(user_bio)

    # Preprocess and combine story fields (title + text if available)
    processed_stories = []
    for story in stories:
        title = story.get('title', '')
        text = story.get('text', '')
        combined_text = title + ' ' + text
        processed_text = preprocess_text(combined_text)
        processed_stories.append({
            "title": title,
            "text": processed_text,
            "url": story.get('url', ''),
            "id": story.get('id', ''),
        })

    # Create a list of texts for embedding
    texts = [user_bio_processed] + [story["text"] for story in processed_stories]

    # Generate embeddings
    embeddings = model.encode(texts)

    # Calculate cosine similarities
    user_bio_embedding = embeddings[0]
    story_embeddings = embeddings[1:]
    cosine_similarities = cosine_similarity([user_bio_embedding], story_embeddings).flatten()

    # Combine stories with their similarity scores
    for i, story in enumerate(processed_stories):
        story["similarity"] = cosine_similarities[i]

    # Sort stories by similarity in descending order
    ranked_stories = sorted(processed_stories, key=lambda x: x["similarity"], reverse=True)

    return ranked_stories

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hacker News Story Ranker</title>
        </head>
        <body>
            <h1>Hacker News Story Ranker</h1>
            <form action="/rank_stories" method="post">
                <label for="bio">Enter your bio:</label><br>
                <textarea id="bio" name="bio" rows="10" cols="50"></textarea><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
    ''')

@app.route('/rank_stories', methods=['POST'])
def rank_stories_endpoint():
    start_time = time.time()
    user_bio = request.form.get('bio', '')
    if not user_bio:
        return jsonify({"error": "User bio is required"}), 400

    print("Fetching top stories...")
    top_story_ids = get_top_stories()
    print(f"Fetched {len(top_story_ids)} top story IDs.")

    stories = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_story_id = {
            executor.submit(get_story_details, story_id): story_id for story_id in top_story_ids
        }
        for future in as_completed(future_to_story_id):
            story_details = future.result()
            if 'title' in story_details and ('url' in story_details or 'text' in story_details):
                stories.append(story_details)
            if len(stories) >= 500:
                break

    print(f"Fetched details for {len(stories)} stories.")

    print("Ranking stories...")
    ranked_stories = rank_stories(user_bio, stories)
    print(f"Ranked {len(ranked_stories)} stories.")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ranked Stories</title>
        </head>
        <body>
            <h1>Ranked Stories</h1>
            <ul>
                {% for story in stories %}
                    <li>
                        <a href="{{ story.url or '#' }}" target="_blank">{{ story.title }}</a>
                        (Similarity Score: {{ story.similarity | round(3) }})
                    </li>
                {% endfor %}
            </ul>
            <a href="/">Back to Home</a>
        </body>
        </html>
    ''', stories=ranked_stories)

if __name__ == '__main__':
    app.run(debug=True)
