from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# Sample Sustainable Products Dataset
data = [
    {"name": "Bamboo Toothbrush", "category": "Personal Care", "price": 9.99, "ecoFriendly": 1, "recyclable": 1, "vegan": 1, "biodegradable": 1},
    {"name": "Reusable Glass Bottle", "category": "Home Goods", "price": 15.99, "ecoFriendly": 1, "recyclable": 1, "vegan": 1, "biodegradable": 0},
    {"name": "Organic Cotton T-Shirt", "category": "Fashion", "price": 19.99, "ecoFriendly": 1, "recyclable": 0, "vegan": 1, "biodegradable": 1},
    {"name": "Solar-Powered Phone Charger", "category": "Electronics", "price": 39.99, "ecoFriendly": 1, "recyclable": 1, "vegan": 0, "biodegradable": 0},
]

df = pd.DataFrame(data)

# Feature Engineering (Combining Text Features for TF-IDF)
df["features"] = df.apply(lambda x: f"{x['category']} eco-{x['ecoFriendly']} recycle-{x['recyclable']} vegan-{x['vegan']} bio-{x['biodegradable']}", axis=1)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["features"])

# Train Model
model = NearestNeighbors(n_neighbors=2, metric="cosine")
model.fit(X)

# Save Model
with open("recommendation_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, df), f)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    input_text = f"{data['category']} eco-{data['ecoFriendly']} recycle-{data['recyclable']} vegan-{data['vegan']} bio-{data['biodegradable']}"
    input_vector = vectorizer.transform([input_text])

    # Find Similar Products
    distances, indices = model.kneighbors(input_vector)
    recommended_products = df.iloc[indices[0]].to_dict(orient="records")

    return jsonify(recommended_products)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
