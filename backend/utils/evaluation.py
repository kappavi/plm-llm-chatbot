import numpy as np
from typing import List, Dict, Any, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

class Evaluator:
    """
    Evaluation module for the PLM-LLM Chatbot
    Implements quantitative and qualitative analysis methods
    """

    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize the evaluator with optional reference data
        """
        self.reference_data = {}
        if reference_data_path and os.path.exists(reference_data_path):
            with open(reference_data_path, 'r') as f:
                self.reference_data = json.load(f)

        # Initialize metrics storage
        self.accuracy_scores = []
        self.bleu_scores = []
        self.similarity_scores = []
        self.benchmark_results = {
            'specialized': [],
            'baseline': []
        }

    def calculate_accuracy(self, prediction: str, reference: str) -> float:
        """
        Calculate accuracy based on keyword matching
        This is a simple implementation that could be improved
        """
        # Extract key terms from reference
        reference_terms = set(reference.lower().split())
        # Extract terms from prediction
        prediction_terms = set(prediction.lower().split())

        # Calculate overlap
        if len(reference_terms) == 0:
            return 0.0

        overlap = reference_terms.intersection(prediction_terms)
        accuracy = len(overlap) / len(reference_terms)

        self.accuracy_scores.append(accuracy)
        return accuracy

    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score for linguistic quality assessment
        """
        # Tokenize
        reference_tokens = [reference.lower().split()]
        prediction_tokens = prediction.lower().split()

        # Use smoothing function to handle edge cases
        smoothie = SmoothingFunction().method1

        # Calculate BLEU
        bleu = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)

        self.bleu_scores.append(bleu)
        return bleu

    def calculate_embedding_similarity(self,
                                      prediction_embedding: np.ndarray,
                                      reference_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings
        """
        # Reshape if needed
        if prediction_embedding.ndim == 1:
            prediction_embedding = prediction_embedding.reshape(1, -1)
        if reference_embedding.ndim == 1:
            reference_embedding = reference_embedding.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(prediction_embedding, reference_embedding)[0][0]

        self.similarity_scores.append(similarity)
        return similarity

    def compare_with_baseline(self,
                             specialized_responses: List[str],
                             baseline_responses: List[str],
                             references: List[str]) -> Dict[str, Any]:
        """
        Compare specialized model with baseline using statistical tests
        """
        specialized_bleu = []
        baseline_bleu = []

        # Calculate BLEU scores for both models
        for i in range(len(references)):
            specialized_bleu.append(self.calculate_bleu(specialized_responses[i], references[i]))
            baseline_bleu.append(self.calculate_bleu(baseline_responses[i], references[i]))

        # Store results
        self.benchmark_results['specialized'] = specialized_bleu
        self.benchmark_results['baseline'] = baseline_bleu

        # Perform t-test
        t_stat, p_value = ttest_ind(specialized_bleu, baseline_bleu)

        return {
            'specialized_mean': np.mean(specialized_bleu),
            'baseline_mean': np.mean(baseline_bleu),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }

    def generate_visualizations(self, output_dir: str):
        """
        Generate visualization plots for evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Accuracy distribution
        if self.accuracy_scores:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.accuracy_scores, kde=True)
            plt.title('Distribution of Accuracy Scores')
            plt.xlabel('Accuracy')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'accuracy_distribution.png'))
            plt.close()

        # BLEU score distribution
        if self.bleu_scores:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.bleu_scores, kde=True)
            plt.title('Distribution of BLEU Scores')
            plt.xlabel('BLEU Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'bleu_distribution.png'))
            plt.close()

        # Similarity score distribution
        if self.similarity_scores:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.similarity_scores, kde=True)
            plt.title('Distribution of Embedding Similarity Scores')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
            plt.close()

        # Benchmark comparison
        if self.benchmark_results['specialized'] and self.benchmark_results['baseline']:
            plt.figure(figsize=(10, 6))
            data = pd.DataFrame({
                'Specialized': self.benchmark_results['specialized'],
                'Baseline': self.benchmark_results['baseline']
            })
            sns.boxplot(data=data)
            plt.title('Comparison of Specialized vs Baseline Model')
            plt.ylabel('BLEU Score')
            plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
            plt.close()

    def save_evaluation_results(self, output_path: str):
        """
        Save evaluation results to a JSON file
        """
        results = {
            'accuracy_scores': self.accuracy_scores,
            'bleu_scores': self.bleu_scores,
            'similarity_scores': self.similarity_scores,
            'benchmark_results': self.benchmark_results,
            'summary': {
                'mean_accuracy': np.mean(self.accuracy_scores) if self.accuracy_scores else 0,
                'mean_bleu': np.mean(self.bleu_scores) if self.bleu_scores else 0,
                'mean_similarity': np.mean(self.similarity_scores) if self.similarity_scores else 0
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_user_feedback(self, feedback_path: str):
        """
        Load and analyze user feedback data
        """
        if not os.path.exists(feedback_path):
            return {'error': 'Feedback file not found'}

        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)

        # Process feedback data
        # This is a placeholder for actual feedback analysis
        ratings = [item.get('rating', 0) for item in feedback_data]

        return {
            'mean_rating': np.mean(ratings) if ratings else 0,
            'total_feedback': len(feedback_data),
            'ratings_distribution': {
                'excellent': len([r for r in ratings if r >= 4.5]),
                'good': len([r for r in ratings if 3.5 <= r < 4.5]),
                'average': len([r for r in ratings if 2.5 <= r < 3.5]),
                'poor': len([r for r in ratings if 1.5 <= r < 2.5]),
                'very_poor': len([r for r in ratings if r < 1.5])
            }
        }

    def generate_error_analysis_report(self,
                                      questions: List[str],
                                      predictions: List[str],
                                      references: List[str],
                                      output_path: str):
        """
        Generate detailed error analysis report
        """
        results = []

        for i in range(len(questions)):
            accuracy = self.calculate_accuracy(predictions[i], references[i])
            bleu = self.calculate_bleu(predictions[i], references[i])

            error_category = 'Unknown'
            if accuracy < 0.3:
                if 'protein' in questions[i].lower():
                    error_category = 'Protein Misidentification'
                elif 'function' in questions[i].lower():
                    error_category = 'Function Misunderstanding'
                else:
                    error_category = 'Factual Error'

            results.append({
                'question': questions[i],
                'prediction': predictions[i],
                'reference': references[i],
                'accuracy': accuracy,
                'bleu': bleu,
                'error_category': error_category if accuracy < 0.5 else 'Correct'
            })

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Also return a summary
        error_categories = [r['error_category'] for r in results if r['error_category'] != 'Correct']
        category_counts = {}
        for category in error_categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            'total_questions': len(questions),
            'correct_answers': len([r for r in results if r['error_category'] == 'Correct']),
            'error_distribution': category_counts
        }

    def visualize_attention(self,
                           text: str,
                           attention_weights: np.ndarray,
                           output_path: str):
        """
        Create heat map visualization of attention weights
        """
        tokens = text.split()

        # Ensure dimensions match
        if len(tokens) != attention_weights.shape[1]:
            # Trim or pad as needed
            if len(tokens) > attention_weights.shape[1]:
                tokens = tokens[:attention_weights.shape[1]]
            else:
                # Pad attention weights
                padding = np.zeros((attention_weights.shape[0],
                                   len(tokens) - attention_weights.shape[1]))
                attention_weights = np.hstack([attention_weights, padding])

        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=False, cmap='viridis')
        plt.title('Attention Visualization')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()