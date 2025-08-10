#!/usr/bin/env python3
"""
Convert inference results to format expected by official HotpotQA evaluation script.
"""

import json
import argparse

def convert_inference_to_official_format(inference_file, output_file):
    """Convert inference results to official HotpotQA evaluation format"""
    
    # Load inference results
    with open(inference_file, 'r') as f:
        inference_results = json.load(f)
    
    # Load original HotpotQA data to get ground truth supporting facts
    with open('hotpot/hotpot_dev_distractor_v1.json', 'r') as f:
        original_data = json.load(f)
    
    # Create mapping from example_id to original data
    id_to_example = {ex['_id']: ex for ex in original_data}
    
    # Initialize output format
    official_format = {
        'answer': {},
        'sp': {}
    }
    
    # Convert each result
    for result in inference_results:
        example_id = result['example_id']
        
        # Add answer
        official_format['answer'][example_id] = result['answer']
        
        # Convert supporting facts format
        # Original format: [paragraph_idx, sentence_idx]
        # We need to convert paragraph indices to [title, sentence_idx] format
        
        if example_id in id_to_example:
            example = id_to_example[example_id]
            context = example['context']
            
            # Convert pred_chain to supporting facts format
            supporting_facts = []
            for para_idx in result['pred_chain'][:3]:  # Top 3 only
                if para_idx < len(context):
                    title = context[para_idx][0]  # Paragraph title
                    # For now, assume sentence 0 (first sentence) - you could improve this
                    supporting_facts.append([title, 0])
            
            official_format['sp'][example_id] = supporting_facts
        else:
            # Fallback if example not found
            official_format['sp'][example_id] = []
    
    # Save converted format
    with open(output_file, 'w') as f:
        json.dump(official_format, f, indent=2)
    
    print(f"Converted {len(inference_results)} examples to official format")
    print(f"Output saved to: {output_file}")
    
    # Print sample conversion
    print("\nSample conversion:")
    sample_id = list(official_format['answer'].keys())[0]
    print(f"Example ID: {sample_id}")
    print(f"Answer: {official_format['answer'][sample_id]}")
    print(f"Supporting Facts: {official_format['sp'][sample_id]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_file", default="inference_improved_full_dev.json", 
                       help="Inference results file")
    parser.add_argument("--output_file", default="official_eval_format.json", 
                       help="Output file for official evaluation")
    
    args = parser.parse_args()
    convert_inference_to_official_format(args.inference_file, args.output_file) 