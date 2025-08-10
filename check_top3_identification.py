#!/usr/bin/env python3
"""
Check if the inference system correctly identifies paragraphs in the TOP 3 positions.
"""

import json
import argparse
from typing import List, Dict, Any

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_top3_identification(example: Dict[str, Any], inference_result: Dict[str, Any]) -> Dict:
    """Analyze if the system identifies paragraphs correctly in top 3 positions"""
    
    # Get ground truth paragraph titles
    gt_titles = set(sf[0] for sf in example['supporting_facts'])
    
    # Get predicted paragraph indices (TOP 3 ONLY)
    predicted_chain = inference_result['pred_chain'][:3]  # Only top 3!
    predicted_titles = []
    valid_predictions = []
    
    for para_idx in predicted_chain:
        if para_idx < len(example['context']):
            title = example['context'][para_idx][0]
            predicted_titles.append(title)
            valid_predictions.append(para_idx)
        else:
            predicted_titles.append(f"INVALID_INDEX_{para_idx}")
    
    # Check if predicted titles match ground truth
    correctly_identified = []
    for i, title in enumerate(predicted_titles):
        if title in gt_titles:
            correctly_identified.append({
                'index': valid_predictions[i] if i < len(valid_predictions) else -1,
                'title': title,
                'position_in_chain': i
            })
    
    # Calculate metrics
    num_gt = len(gt_titles)
    num_correct = len(correctly_identified)
    num_predicted = len(predicted_chain)
    
    # Coverage: what fraction of ground truth we found
    coverage = num_correct / num_gt if num_gt > 0 else 0
    
    # Precision: what fraction of predictions were correct
    precision = num_correct / num_predicted if num_predicted > 0 else 0
    
    # F1 Score: harmonic mean of precision and recall
    f1_score = 2 * (precision * coverage) / (precision + coverage) if (precision + coverage) > 0 else 0
    
    return {
        'gt_titles': list(gt_titles),
        'predicted_titles': predicted_titles,
        'correctly_identified': correctly_identified,
        'num_gt': num_gt,
        'num_correct': num_correct,
        'num_predicted': num_predicted,
        'coverage': coverage,  # recall
        'precision': precision,
        'f1_score': f1_score,
        'is_perfect': num_correct == num_gt,  # Perfect if we found ALL GT paragraphs (order doesn't matter)
        'is_exact_match': num_correct == num_gt and num_correct == num_predicted  # Exact position match
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_file", default="inference_improved_full_dev.json", help="Inference results file")
    parser.add_argument("--original_data", default="hotpot/hotpot_dev_distractor_v1.json", help="Original data file")
    parser.add_argument("--full_evaluation", action="store_true", help="Evaluate on full dataset")
    args = parser.parse_args()
    
    # Load data
    original_data = load_data(args.original_data)
    inference_results = load_data(args.inference_file)
    
    print("TOP-3 PARAGRAPH IDENTIFICATION EVALUATION - FULL DEV DATASET")
    print("=" * 65)
    
    # Create mapping
    id_to_example = {ex['_id']: ex for ex in original_data}
    
    total_examples = 0
    total_coverage = 0
    total_precision = 0
    total_f1 = 0
    perfect_matches = 0  # Found ALL GT paragraphs (order doesn't matter)
    exact_position_matches = 0  # Found ALL GT paragraphs in exact positions
    full_coverage = 0  # Found all GT paragraphs
    
    for result in inference_results:
        example_id = result['example_id']
        if example_id in id_to_example:
            example = id_to_example[example_id]
            analysis = analyze_top3_identification(example, result)
            
            total_examples += 1
            total_coverage += analysis['coverage']
            total_precision += analysis['precision']
            total_f1 += analysis['f1_score']
            
            if analysis['is_perfect']:
                perfect_matches += 1
            
            if analysis['is_exact_match']:
                exact_position_matches += 1
            
            if analysis['coverage'] == 1.0:  # Found all GT paragraphs
                full_coverage += 1
            
            # Only print first few examples to avoid overwhelming output
            if total_examples <= 5:
                print(f"\nExample ID: {example_id}")
                print(f"Question: {example['question'][:80]}...")
                print(f"Ground Truth ({analysis['num_gt']}): {analysis['gt_titles']}")
                print(f"Top-3 Chain: {result['pred_chain'][:3]}")
                print(f"Top-3 Titles: {analysis['predicted_titles']}")
                print(f"Correctly Identified: {[item['title'] for item in analysis['correctly_identified']]}")
                print(f"Coverage: {analysis['coverage']:.1%} ({analysis['num_correct']}/{analysis['num_gt']})")
                print(f"Precision: {analysis['precision']:.1%} ({analysis['num_correct']}/{analysis['num_predicted']})")
                print(f"F1 Score: {analysis['f1_score']:.1%}")
                
                if analysis['is_exact_match']:
                    print("STATUS: EXACT POSITION MATCH")
                elif analysis['is_perfect']:
                    print("STATUS: PERFECT MATCH (found all GT paragraphs, order doesn't matter)")
                elif analysis['coverage'] == 1.0:
                    print("STATUS: FULL COVERAGE (found all GT paragraphs)")
                elif analysis['coverage'] >= 0.5:
                    print("STATUS: GOOD COVERAGE")
                else:
                    print("STATUS: POOR COVERAGE")
    
    print(f"\nFULL DEV DATASET STATISTICS (TOP-3 EVALUATION):")
    print(f"   Total examples: {total_examples}")
    print(f"   Perfect matches (found all GT): {perfect_matches}/{total_examples} ({perfect_matches/total_examples:.1%})")
    print(f"   Exact position matches: {exact_position_matches}/{total_examples} ({exact_position_matches/total_examples:.1%})")
    print(f"   Full coverage: {full_coverage}/{total_examples} ({full_coverage/total_examples:.1%})")
    print(f"   Average coverage: {total_coverage/total_examples:.1%}")
    print(f"   Average precision: {total_precision/total_examples:.1%}")
    print(f"   Average F1 Score: {total_f1/total_examples:.1%}")
    
    print(f"\nPERFORMANCE ANALYSIS:")
    avg_coverage = total_coverage/total_examples if total_examples > 0 else 0
    avg_precision = total_precision/total_examples if total_examples > 0 else 0
    avg_f1 = total_f1/total_examples if total_examples > 0 else 0
    
    print(f"   Overall Performance: {avg_coverage:.1%} coverage, {avg_precision:.1%} precision, {avg_f1:.1%} F1")
    
    if avg_coverage > 0.7 and avg_precision > 0.7:
        print(f"   RESULT: EXCELLENT - High coverage AND precision in top-3")
        print(f"   CONCLUSION: System successfully identifies relevant paragraphs")
    elif avg_coverage > 0.5:
        print(f"   RESULT: GOOD - Decent coverage, system finds most relevant paragraphs")
    else:
        print(f"   RESULT: POOR - Low coverage, missing too many relevant paragraphs")
    
    if avg_f1 > 0.6:
        print(f"   F1 PERFORMANCE: STRONG - Balanced precision and recall performance")
    elif avg_f1 > 0.4:
        print(f"   F1 PERFORMANCE: MODERATE - Room for improvement in balance")
    else:
        print(f"   F1 PERFORMANCE: WEAK - Significant imbalance between precision and recall")

if __name__ == "__main__":
    main() 