import json
from typing import List, Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())

def syntactic_similarity(aspect: str, field_value: str) -> float:
    aspect_proc = preprocess_text(aspect)
    field_proc = preprocess_text(field_value)
    if not aspect_proc or not field_proc:
        return 0.0
        
    vectorizer = CountVectorizer().fit([aspect_proc, field_proc])
    vectors = vectorizer.transform([aspect_proc, field_proc])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def load_registry(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {line.strip()}")
    except FileNotFoundError:
        print(f"Registry file not found: {file_path}")
    return data

def match_services(registry: List[Dict], aspects: Dict[str, str]) -> List[Dict]:
    min_threshold = 0.3
    scored_entries = []

    for entry in registry:
        total_score = 0.0
        matched_aspects = 0
        for aspect_key, aspect_value in aspects.items():
            if aspect_key in entry and entry[aspect_key]:
                sim = syntactic_similarity(aspect_value, entry[aspect_key])
                if sim >= min_threshold:
                    total_score += sim
                    matched_aspects += 1
        if matched_aspects > 0:
            scored_entries.append({"score": total_score, "service": entry})

    if not scored_entries:
        return []

    scored_entries.sort(key=lambda x: x["score"], reverse=True)
    return [item["service"] for item in scored_entries]

class SyntacticModelWrapper:
    def __init__(self, registry_path: str):
        if not os.path.exists(registry_path):
             raise FileNotFoundError(f"The registry file was not found at {registry_path}")
        self.registry = load_registry(registry_path)
        if not self.registry:
            raise ValueError("Failed to load or registry is empty.")
        print("SyntacticModelWrapper initialized and registry loaded.")

    def predict(self, query: str) -> Dict:
        if not query:
            return {}
            
        aspects = {
            "func_name": query,
            "docstring": query 
        }
        
        top_matches = match_services(self.registry, aspects)
        
        if top_matches:
            return top_matches[0] 
        else:
            return {}

if __name__ == '__main__':
    try:
        wrapper = SyntacticModelWrapper(registry_path='dummy_registry.jsonl')
        test_query = "user authentication"
        prediction = wrapper.predict(test_query)
        
        print(f"\nQuery: '{test_query}'")
        print("Prediction:")
        print(json.dumps(prediction, indent=2))
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please create a file named 'dummy_registry.jsonl' in the 'servio-evaluation' directory and add some sample data to it.")