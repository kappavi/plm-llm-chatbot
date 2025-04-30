import os
from openai import OpenAI
from typing import Dict, List, Optional

class LLMIntegration:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"  # or "gpt-3.5-turbo" for faster responses
        
    def get_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Get chat completion from OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the content from the response
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
            
    def format_response(self, question: str, protein_context: Optional[Dict] = None, history: Optional[List[Dict]] = None) -> str:
        """
        Format the response using protein context and conversation history
        """
        system_prompt = """You are a biology expert assistant specialized in answering questions about biology, 
        including proteins, DNA, RNA, and other biological molecules. 
        When analyzing protein sequences, you should:
        1. Identify the protein based on its sequence
        2. Describe its structure and function
        3. Explain any notable features or domains
        4. Provide relevant biological context
        
        For protein-related questions, use the provided protein context to inform your answer.
        For general biology questions, provide accurate information based on your expertise.
        
        When responding to follow-up questions, maintain context from previous messages and provide
        additional details or clarification as needed."""
        
        # Initialize protein_context if None
        if protein_context is None:
            protein_context = {}
            
        # Initialize history if None
        if history is None:
            history = []
            
        # Check if we have a protein sequence in the context
        if protein_context and protein_context.get('sequence'):
            user_prompt = f"""Question: {question}
            
            Protein Context:
            - Sequence: {protein_context['sequence']}
            
            Please analyze this protein sequence and provide a detailed answer to the question.
            Focus on the specific protein identified by this sequence, not general information about similar proteins.
            If you recognize this sequence, provide specific details about this particular protein."""
        else:
            user_prompt = f"""Question: {question}
            
            Please provide a detailed answer based on your knowledge."""
        
        # Build messages list with system prompt, history, and current question
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(history)
        
        # Add current question
        messages.append({"role": "user", "content": user_prompt})
        
        return self.get_chat_completion(messages) 