#!/usr/bin/env python
import argparse
import json
import os
from models.plm_integration import ProteinModel
from models.llm_integration import LLMIntegration
from utils.evaluation import Evaluator
import numpy as np
from tqdm import tqdm

def load_test_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_evaluation(args):
    print(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)

    print("Initializing models...")
    protein_model = ProteinModel()
    llm_model = LLMIntegration()

    evaluator = Evaluator()

    os.makedirs(args.output_dir, exist_ok=True)

    questions = []
    predictions = []
    references = []

    print(f"Evaluating on {len(test_data)} test examples...")
    for example in tqdm(test_data):
        question = example['question']
        reference = example['reference']

        protein_context = protein_model.get_protein_context(question)

        if 'embeddings' in protein_context and protein_context['embeddings'] is not None:
            embeddings = protein_context['embeddings']
        else:
            embeddings = None

        prediction = llm_model.format_response(question=question, protein_context=protein_context)

        questions.append(question)
        predictions.append(prediction)
        references.append(reference)

        if 'reference_embeddings' in example and embeddings is not None:
            reference_embeddings = np.array(example['reference_embeddings'])
            evaluator.calculate_embedding_similarity(embeddings, reference_embeddings)

    print("Generating error analysis report...")
    error_summary = evaluator.generate_error_analysis_report(
        questions=questions,
        predictions=predictions,
        references=references,
        output_path=os.path.join(args.output_dir, 'error_analysis.json')
    )

    print("Generating visualizations...")
    evaluator.generate_visualizations(args.output_dir)

    print("Saving evaluation results...")
    evaluator.save_evaluation_results(os.path.join(args.output_dir, 'evaluation_results.json'))

    print("\nEvaluation Summary:")
    print(f"Total questions: {error_summary['total_questions']}")
    print(f"Correct answers: {error_summary['correct_answers']}")
    print(f"Accuracy: {error_summary['correct_answers'] / error_summary['total_questions']:.2f}")
    print("\nError Distribution:")
    for category, count in error_summary['error_distribution'].items():
        print(f"  {category}: {count}")

    print(f"\nDetailed results saved to {args.output_dir}")

def compare_with_baseline(args):
    print(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data)

    print(f"Loading baseline results from {args.baseline_results}")
    with open(args.baseline_results, 'r') as f:
        baseline_results = json.load(f)

    print("Initializing specialized models...")
    protein_model = ProteinModel()
    llm_model = LLMIntegration()

    evaluator = Evaluator()

    os.makedirs(args.output_dir, exist_ok=True)

    specialized_responses = []
    baseline_responses = []
    references = []

    print(f"Evaluating on {len(test_data)} test examples...")
    for i, example in enumerate(tqdm(test_data)):
        if i >= len(baseline_results):
            break

        question = example['question']
        reference = example['reference']
        baseline_response = baseline_results[i]['response']

        protein_context = protein_model.get_protein_context(question)

        specialized_response = llm_model.format_response(question=question, protein_context=protein_context)

        specialized_responses.append(specialized_response)
        baseline_responses.append(baseline_response)
        references.append(reference)

    print("Comparing models...")
    comparison_results = evaluator.compare_with_baseline(
        specialized_responses=specialized_responses,
        baseline_responses=baseline_responses,
        references=references
    )

    print("Generating visualizations...")
    evaluator.generate_visualizations(args.output_dir)

    print("Saving evaluation results...")
    evaluator.save_evaluation_results(os.path.join(args.output_dir, 'comparison_results.json'))

    print("\nComparison Summary:")
    print(f"Specialized model mean BLEU score: {comparison_results['specialized_mean']:.4f}")
    print(f"Baseline model mean BLEU score: {comparison_results['baseline_mean']:.4f}")
    print(f"T-statistic: {comparison_results['t_statistic']:.4f}")
    print(f"P-value: {comparison_results['p_value']:.4f}")
    print(f"Statistically significant difference: {comparison_results['significant_difference']}")

    print(f"\nDetailed results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the PLM-LLM Chatbot')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation on test dataset')
    eval_parser.add_argument('--test-data', required=True, help='Path to test dataset JSON file')
    eval_parser.add_argument('--output-dir', default='evaluation_results', help='Directory to save results')

    compare_parser = subparsers.add_parser('compare', help='Compare with baseline LLM')
    compare_parser.add_argument('--test-data', required=True, help='Path to test dataset JSON file')
    compare_parser.add_argument('--baseline-results', required=True, help='Path to baseline results JSON file')
    compare_parser.add_argument('--output-dir', default='comparison_results', help='Directory to save results')

    args = parser.parse_args()

    if args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'compare':
        compare_with_baseline(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()