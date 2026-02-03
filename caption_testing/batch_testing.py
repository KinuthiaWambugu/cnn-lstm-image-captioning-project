"""
Batch testing on multiple images
"""

import tensorflow as tf
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

from test_image_captioning import(
    load_model,
    generate_caption,
    MAX_LENGTH
)

def batch_generate_captions(images_dir, caption_model, tokenizer, temperature=1.0, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', ',png', '.gif']
    
    images_dir = Path(images_dir)
    images_files = []
    for ext in extensions:
        images_files.extend(images_dir.glob(f'*{ext}'))
        images_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    images_files = sorted(set(images_files))
    print(f"Found {len(images_files)} images to process")
    results = {}

    for img_path in tqdm(images_files, desc="Generating captions"):
        try:
            caption = generate_caption(
                str(img_path), caption_model, tokenizer, temperature=temperature
            )
            results[str(img_path)] = {
                'caption': caption,
                'status': 'success'
            }
        except Exception as e:
            print(f"\n Error processing {img_path}: {e}")
            results[str(img_path)] = {
                'caption':None,
                'status': 'error',
                'error': str(e)
            }
    return results

def save_results(results, output_path, include_stats=True):
    output_data = {
        'results': results
    }
    
    if include_stats:
        total = len(results)
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        failed = total - successful
        
        # Calculate average caption length
        captions = [r['caption'] for r in results.values() if r['caption']]
        avg_length = np.mean([len(c.split()) for c in captions]) if captions else 0
        
        output_data['statistics'] = {
            'total_images': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'average_caption_length': float(avg_length)
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_results_summary(results):
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for img_path, result in results.items():
        img_name = Path(img_path).name
        if result['status'] == 'success':
            print(f"\n{img_name}")
            print(f"  → {result['caption']}")
        else:
            print(f"\n{img_name}")
            print(f"  → ERROR: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate captions for multiple images in a directory'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing images to caption'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights file'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='caption_results.json',
        help='Output JSON file for results (default: caption_results.json)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature for greedy decoding (default: 1.0)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.jpg', '.jpeg', '.png'],
        help='Image file extensions to process (default: .jpg .jpeg .png)'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Do not print results summary to console'
    )
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not Path(args.images_dir).exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        return
    if not Path(args.vocab).exists():
        print(f"Error: Vocabulary file not found: {args.vocab}")
        return
    
    # Load model
    print("Loading model...")
    caption_model, tokenizer = load_model(args.weights, args.vocab)
    
    # Generate captions
    print(f"\nProcessing images from: {args.images_dir}")
    results = batch_generate_captions(
        args.images_dir,
        caption_model,
        tokenizer,
        temperature=args.temperature,
        extensions=args.extensions
    )
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    if not args.no_summary:
        print_results_summary(results)


if __name__ == "__main__":
    main()