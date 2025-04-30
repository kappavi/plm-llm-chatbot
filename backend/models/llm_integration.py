import os
from openai import OpenAI
from typing import Dict, List, Optional

class LLMHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"  # Using GPT-4 for better performance
        
    def generate_response(self, question: str, context: Dict) -> str:
        """
        Generate a response using ChatGPT with the provided context
        """
        # Construct the prompt with context
        prompt = f"""You are a biology expert assistant specializing in protein structure and function.
        Use the following context to answer the question accurately and professionally.

        Context:
        Protein Name: {context.get('protein_name', 'Unknown')}
        Sequence: {context.get('sequence', 'Not provided')}
        Additional Information: {context.get('additional_info', 'None')}

        Question: {question}

        Please provide a detailed, accurate response based on the context and your knowledge.
        If the context is insufficient, please indicate that and suggest what additional information might be needed.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a biology expert assistant specializing in protein structure and function."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}" 