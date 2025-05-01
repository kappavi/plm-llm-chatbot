from flask import Flask, request, jsonify
from flask_cors import CORS
from models.plm_integration import ProteinModel
from models.llm_integration import LLMIntegration
from utils.feedback import FeedbackManager
import os
from dotenv import load_dotenv
import numpy as np
from typing import Dict, List

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize models
try:
    protein_model = ProteinModel()
    llm_model = LLMIntegration()
    feedback_manager = FeedbackManager()
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)

# Store conversation history in memory (in production, use a proper database)
conversation_history: Dict[str, List[Dict]] = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        conversation_id = data.get('conversation_id', 'default')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Get protein context
        protein_context = protein_model.get_protein_context(question)

        # Convert numpy arrays to lists for JSON serialization
        if 'embeddings' in protein_context and protein_context['embeddings'] is not None:
            protein_context['embeddings'] = protein_context['embeddings'].tolist()

        # Get conversation history
        history = conversation_history.get(conversation_id, [])

        # Get response from LLM with conversation history
        response = llm_model.format_response(question=question, protein_context=protein_context, history=history)

        # Update conversation history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        conversation_history[conversation_id] = history

        return jsonify({
            'response': response,
            'protein_context': protein_context,
            'conversation_id': conversation_id
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json

        # Extract feedback data
        question = data.get('question', '')
        response = data.get('response', '')
        rating = data.get('rating')
        comments = data.get('comments')
        user_id = data.get('user_id')

        # Validate required fields
        if not question or not response or rating is None:
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate rating
        try:
            rating = int(rating)
            if not 1 <= rating <= 5:
                return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid rating format'}), 400

        # Add feedback
        feedback_id = feedback_manager.add_feedback(
            question=question,
            response=response,
            rating=rating,
            comments=comments,
            user_id=user_id
        )

        return jsonify({
            'status': 'success',
            'feedback_id': feedback_id
        })

    except Exception as e:
        print(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/summary', methods=['GET'])
def get_feedback_summary():
    try:
        summary = feedback_manager.get_feedback_summary()
        return jsonify(summary)
    except Exception as e:
        print(f"Error in feedback summary endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=True)