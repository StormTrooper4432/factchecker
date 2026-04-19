from typing import List


KEYWORD_TAGS = {
    "vegan": ["vegan", "plant-based", "plant based"],
    "vegetarian": ["vegetarian", "lacto-ovo"],
    "omega-3": ["omega-3", "dha", "epa", "fish oil", "omega 3"],
    "vitamin b12": ["b12", "cobalamin"],
    "supplement": ["supplement", "capsule", "tablet"],
    "fish": ["fish", "salmon", "tuna", "mackerel", "trout"],
    "dairy": ["milk", "cheese", "yogurt"],
    "ultra-processed": ["ultra-processed", "processed food", "packaged snack"],
    "cholesterol": ["cholesterol"],
    "triglyceride": ["triglyceride", "triglycerides"],
}


def generate_tags(text: str) -> List[str]:
    lower = text.lower()
    tags = []
    for tag, needles in KEYWORD_TAGS.items():
        if any(n in lower for n in needles):
            tags.append(tag)
    # ensure at least one tag
    return tags or ["untagged"]
