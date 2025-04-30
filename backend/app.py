from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from models.plm_integration import ProteinModel
from models.llm_integration import LLMHandler
from utils.retriever import VectorRetriever

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

app = Flask(__name__)
CORS(app)

# Initialize models
try:
    protein_model = ProteinModel()
    llm_handler = LLMHandler()
    vector_retriever = VectorRetriever()
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get protein embeddings and context
        protein_context = protein_model.get_protein_context(question)
        
        # Retrieve relevant information
        relevant_info = vector_retriever.retrieve(question, protein_context)
        
        # Generate response using LLM
        response = llm_handler.generate_response(question, relevant_info)
        
        return jsonify({
            'response': response,
            'context': relevant_info
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting server...")
    print(f"OpenAI API Key is {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    app.run(debug=True, port=6001) 