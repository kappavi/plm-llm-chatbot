from flask import Flask, request, jsonify
from flask_cors import CORS
from models.plm_integration import ProteinModel
from models.llm_integration import LLMIntegration
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize models
try:
    protein_model = ProteinModel()
    llm_model = LLMIntegration()
    print("Models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        # Get protein context
        protein_context = protein_model.get_protein_context(question)
        
        # Convert numpy arrays to lists for JSON serialization
        if 'embeddings' in protein_context:
            protein_context['embeddings'] = protein_context['embeddings'].tolist()
        
        # Get response from LLM
        response = llm_model.format_response(question, protein_context)
        
        return jsonify({
            'response': response,
            'protein_context': protein_context
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=True) 