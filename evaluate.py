import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from model_wrappers.direct_syntactic_wrapper import SyntacticModelWrapper
from model_wrappers.direct_sequencial_wrapper import SequencialModelWrapper
from model_wrappers.parallel_wrapper import ParallelModelWrapper
from model_wrappers.direct_llm_wrapper import LLMModelWrapper

import metrics_calculator

def load_dataset(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def main():
    dataset_path = 'dummy_registry.jsonl'
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    evaluation_data = []
    for service in dataset:
        query = service.get("func_name", "").replace("_", " ").replace("create", "").strip()
        if query:
            evaluation_data.append({"query": query, "expected_service": service})

    print(f"Generated {len(evaluation_data)} evaluation queries.")

    print("Initializing models...")
    models = {
        "syntactic": SyntacticModelWrapper(registry_path=dataset_path),
        "sequencial": SequencialModelWrapper(registry_path=dataset_path),
        "parallel": ParallelModelWrapper(registry_path=dataset_path),
        "llm": LLMModelWrapper(registry_path=dataset_path)
    }
    
    final_results = []

    for model_name, model in models.items():
        print(f"\n--- Testing model: {model_name} ---")
        
        predictions = []
        for item in tqdm(evaluation_data, desc=f"Querying {model_name}"):
            query = item["query"]
            expected_service = item["expected_service"]
            
            predicted_service = model.predict(query)
            predictions.append({
                "query": query,
                "predicted": predicted_service,
                "expected": expected_service
            })
        
        overall_metrics = metrics_calculator.calculate_all_metrics(predictions)
        overall_metrics['model'] = model_name
        overall_metrics['dataset'] = dataset_path
        final_results.append(overall_metrics)
        print(f"Results for {model_name}: {overall_metrics}")

    results_df = pd.DataFrame(final_results)
    results_df.to_csv("evaluation_report.csv", index=False)
    print("\nEvaluation complete. Report saved to evaluation_report.csv")
    print("\nReport content:")
    print(results_df)

if __name__ == "__main__":
    main()
