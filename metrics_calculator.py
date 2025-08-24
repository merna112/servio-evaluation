from typing import List, Dict

def is_match(predicted: Dict, expected: Dict) -> bool:
    if not predicted or not expected:
        return False
    return predicted.get('url') == expected.get('url')

def calculate_precision_at_1(predictions: List[Dict]) -> float:
    correct_predictions = 0
    total_predictions = len(predictions)
    
    if total_predictions == 0:
        return 0.0

    for item in predictions:
        if is_match(item.get('predicted'), item.get('expected')):
            correct_predictions += 1
            
    return correct_predictions / total_predictions

def calculate_all_metrics(predictions: List[Dict]) -> Dict:
    precision_at_1 = calculate_precision_at_1(predictions)
    
    return {
        "precision_at_1": precision_at_1,
        "total_queries": len(predictions)
    }