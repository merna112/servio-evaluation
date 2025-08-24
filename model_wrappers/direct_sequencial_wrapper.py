import json
from typing import List, Dict
import nltk
from nltk.corpus import wordnet as wn
import os

try:
    wn.synsets('dog')
except Exception:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())

def enhanced_similarity(aspect: str, field_value: str) -> float:
    aspect_proc = preprocess_text(aspect)
    field_proc = preprocess_text(field_value)

    if not aspect_proc or not field_proc:
        return 0.0
        
    if aspect_proc in field_proc:
        return 1.0

    synsets_aspect = wn.synsets(aspect_proc)
    synsets_field = wn.synsets(field_proc)
    
    max_sim = 0.0
    if not synsets_aspect or not synsets_field:
        return max_sim

    for syn1 in synsets_aspect:
        for syn2 in synsets_field:
            sim = syn1.wup_similarity(syn2) or 0.0
            if sim > max_sim:
                max_sim = sim
    return max_sim

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
                sim = enhanced_similarity(aspect_value, entry[aspect_key])
                if sim >= min_threshold:
                    total_score += sim
                    matched_aspects += 1
        if matched_aspects > 0:
            scored_entries.append({"score": total_score, "service": entry})

    if not scored_entries:
        return []

    scored_entries.sort(key=lambda x: x["score"], reverse=True)
    return [item["service"] for item in scored_entries]

class SequencialModelWrapper:
    def __init__(self, registry_path: str):
        if not os.path.exists(registry_path):
             raise FileNotFoundError(f"The registry file was not found at {registry_path}")
        self.registry = load_registry(registry_path)
        if not self.registry:
            raise ValueError("Failed to load or registry is empty.")
        print("SequencialModelWrapper initialized and registry loaded.")

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
        wrapper = SequencialModelWrapper(registry_path='dummy_registry.jsonl')
        
        test_query = "process payment"
        prediction = wrapper.predict(test_query)
        
        print(f"\nQuery: '{test_query}'")
        print("Prediction:")
        print(json.dumps(prediction, indent=2))
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure 'dummy_registry.jsonl' exists in the 'servio-evaluation' directory.")