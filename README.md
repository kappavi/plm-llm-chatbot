# PLM-LLM Biology QA Chatbot

A specialized chatbot that combines Protein Language Models (PLMs) with Large Language Models (LLMs) to provide accurate and detailed answers to biology-related questions.

## Features

- **Protein Analysis**: Uses ESM and ProtT5 models to analyze protein sequences and generate embeddings
- **Intelligent Responses**: Combines protein context with general biological knowledge
- **User-Friendly Interface**: Clean, modern chat interface
- **Real-time Processing**: Quick responses with detailed explanations

## System Architecture

### Pipeline Overview

1. **Frontend to Backend**:
   - User types a question in the React frontend
   - Frontend makes a POST request to `http://localhost:6001/api/chat` with the question
   - Request is handled by the Flask backend

2. **Backend Processing**:
   ```python
   @app.route('/api/chat', methods=['POST'])
   def chat():
       # 1. Get question from request
       question = data.get('question', '')
       
       # 2. Get protein context using PLM
       protein_context = protein_model.get_protein_context(question)
       
       # 3. Get response from LLM (ChatGPT)
       response = llm_model.format_response(question, protein_context)
   ```

3. **PLM (Protein Language Model) Processing**:
   - The `protein_model.get_protein_context(question)` method:
     - Checks if the question contains a protein sequence
     - If yes, generates embeddings using ESM/ProtT5 models
     - If no, checks for protein names and queries UniProt
     - Returns protein context (sequence, embeddings, etc.)

4. **LLM (ChatGPT) Processing**:
   - The `llm_model.format_response(question, protein_context)` method:
     - Takes the question and protein context
     - Formats them into a prompt for ChatGPT
     - Sends the prompt to OpenAI's API
     - Returns ChatGPT's response

5. **Response Flow**:
   - Backend combines the response and protein context
   - Sends it back to the frontend
   - Frontend displays the response to the user

### Visual Pipeline
```
User Question
     ↓
Frontend (React)
     ↓
Backend (Flask)
     ↓
PLM (ESM/ProtT5) → Protein Context
     ↓
LLM (ChatGPT) → Final Response
     ↓
Frontend (React)
     ↓
User
```

### Key Points
1. The PLM and LLM don't talk directly to each other
2. The backend orchestrates the flow:
   - First gets protein context from PLM
   - Then uses that context to get a better answer from ChatGPT
3. The PLM is only used when the question is about proteins
4. For general biology questions, only ChatGPT is used

## Project Structure

```
plm-llm-chatbot/
├── backend/
│   ├── app.py                 # Flask application
│   ├── models/
│   │   ├── plm_integration.py # Protein model integration
│   │   └── llm_integration.py # ChatGPT integration
│   ├── utils/
│   │   └── retriever.py       # Vector retrieval utilities
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
└── frontend/
    ├── src/
    │   ├── App.tsx           # Main React component
    │   └── index.tsx         # Entry point
    ├── package.json          # Node.js dependencies
    └── tsconfig.json         # TypeScript configuration
```

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env-sample .env
   # Edit .env with your OpenAI API key
   ```

5. Run the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Technologies Used

### Backend
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for PLMs
- **OpenAI API**: For ChatGPT integration
- **Biopython**: For biological data processing

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Styling
- **Axios**: HTTP client

## Usage Examples

1. **Protein Sequence Analysis**:
   ```
   Q: What can you tell me about this protein sequence: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN?
   ```

2. **Protein Name Query**:
   ```
   Q: Tell me about the protein INSULIN
   ```

3. **General Biology Questions**:
   ```
   Q: What is a nucleotide?
   ```

## Acknowledgments

- ESM and ProtT5 models from Facebook and Rostlab
- OpenAI for ChatGPT
- Hugging Face for the Transformers library
