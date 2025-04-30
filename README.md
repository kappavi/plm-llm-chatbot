# PLM-LLM Biology QA System

A sophisticated Q&A chatbot that combines protein language models (PLMs) with large language models (LLMs) to provide accurate, domain-specific answers to biology questions.

## Features

- Integration of ESM/ProtT5 protein language models with ChatGPT
- Retrieval-augmented generation (RAG) for accurate responses
- Modern React/TypeScript frontend with Tailwind CSS
- Flask backend for model serving and API endpoints
- Real-time chat interface for biology Q&A

## Project Structure

```
plm-llm-chatbot/
├── backend/           # Flask backend
│   ├── app.py        # Main Flask application
│   ├── models/       # PLM and LLM integration
│   ├── utils/        # Utility functions
│   └── requirements.txt
├── frontend/         # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
└── README.md
```

## Setup

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY=your_api_key
export FLASK_APP=app.py
export FLASK_ENV=development
```

4. Run the backend:
```bash
flask run
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the development server:
```bash
npm start
```

## Technologies Used

- Backend:
  - Python 3.8+
  - Flask
  - ESM/ProtT5 protein language models
  - OpenAI GPT-4 API
  - FAISS for vector similarity search

- Frontend:
  - React
  - TypeScript
  - Tailwind CSS
  - Axios for API calls

## License

MIT License
