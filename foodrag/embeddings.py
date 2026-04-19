import numpy as np
from typing import List

# Lightweight, offline-safe embedding tailored to food/nutrition terms.

FOOD_TERMS = [
    "omega-3",
    "dha",
    "epa",
    "fish",
    "salmon",
    "tuna",
    "mackerel",
    "vitamin",
    "b12",
    "cholesterol",
    "triglyceride",
    "vegan",
    "vegetarian",
    "supplement",
    "dairy",
    "protein",
    "fatty",
    "acid",
    "fiber",
    "plant",
    "meat",
    "poultry",
    "omega",
    "lipid",
    # Added for gluten/weight/bmi/dietary patterns
    "gluten",
    "celiac",
    "weight",
    "bmi",
    "body mass",
    "diet",
    "dietary",
    "calorie",
    "loss",
    "obesity",
    "overweight",
    "glycemic",
]


class FoodEmbeddingFunction:
    def name(self) -> str:
        return "food_bow_v1"

    # Chroma expects __call__(self, input: List[str])
    def __call__(self, input: List[str]) -> List[List[float]]:
        vectors = []
        for t in input:
            lower = t.lower()
            counts = [lower.count(term) for term in FOOD_TERMS]
            vec = np.array(counts, dtype=float)
            norm = np.linalg.norm(vec) + 1e-8
            vectors.append((vec / norm).tolist())
        return vectors

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input):
        if isinstance(input, list):
            return self(input)
        return self([input])[0]
