import json
from typing import List, Dict
import os
import getpass
from groq import Groq

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

class LLMModelWrapper:
    def __init__(self, registry_path: str):
        if not os.path.exists(registry_path):
             raise FileNotFoundError(f"The registry file was not found at {registry_path}")
        self.registry = load_registry(registry_path)
        if not self.registry:
            raise ValueError("Failed to load or registry is empty.")

        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                print("Groq API key not found in environment variables.")
                api_key = getpass.getpass("Please enter your Groq API key: ")
            
            self.client = Groq(api_key=api_key)
            print("Groq client initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")
            
        print("LLMModelWrapper initialized and registry loaded.")

    def predict(self, query: str) -> Dict:
        if not query:
            return {}

        services_str = json.dumps(self.registry[:20], indent=2)

        prompt = f"""
        You are an expert service recommender.
        Given a user query and a list of available services in JSON format, your task is to choose the single best service that matches the query.
        You MUST return the result as a single, valid JSON object of the chosen service. Do not add any explanation or introductory text.
        
        User Query: "{query}"
        
        Available Services:
        {services_str}
        
        Best Matching Service (JSON format only):
        """

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            result_json = json.loads(response_content)
            return result_json
            
        except Exception as e:
            print(f"An error occurred while calling the Groq API: {e}")
            return {}

if __name__ == '__main__':
    try:
        wrapper = LLMModelWrapper(registry_path='dummy_registry.jsonl')
        test_query = "a service for user login and security"
        prediction = wrapper.predict(test_query)
        
        print(f"\nQuery: '{test_query}'")
        print("Prediction:")
        print(json.dumps(prediction, indent=2))

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}")