from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trend_data = {
    "dates": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
    "popularity": [10, 20, 30, 40, 50]
}

@app.get("/api/trends")
def get_trends():
    return trend_data

trend_data_np = np.array([
    [10, 20, 30, 40, 50],  # Trend features
])

class RecommendationEngine:
    def __init__(self, trend_data):
        self.trend_data = trend_data

    def recommend_outfits(self, user_preferences):
        model = NearestNeighbors(n_neighbors=5)
        model.fit(self.trend_data)
        distances, indices = model.kneighbors([user_preferences])
        recommended_outfits = self.trend_data[indices[0]]
        return recommended_outfits.tolist()

@app.post("/api/recommendations")
def get_recommendations(user_preferences: list):
    engine = RecommendationEngine(trend_data_np)
    recommendations = engine.recommend_outfits(np.array(user_preferences))
    return {"recommendations": recommendations}

@app.post("/api/customize")
def customize_outfit(customization_options: dict):
    # Handle customization logic
    return {"status": "Customization applied", "options": customization_options}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
