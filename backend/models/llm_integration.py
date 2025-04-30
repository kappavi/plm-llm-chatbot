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
            
    def format_response(self, question: str, protein_context: Dict) -> str:
        """
        Format the response using protein context
        """
        system_prompt = """You are a biology expert assistant specialized in protein-related questions. 
        Use the provided protein context to answer questions accurately and informatively.
        If you don't have enough information, say so."""
        
        user_prompt = f"""Question: {question}
        Protein Context:
        - Name: {protein_context.get('protein_name', 'Unknown')}
        - Sequence: {protein_context.get('sequence', 'Not available')}
        
        Please provide a detailed answer based on this context."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.get_chat_completion(messages) 