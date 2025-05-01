import os
from openai import OpenAI
from typing import Dict, List, Optional

class LLMIntegration:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4.1-nano"

    def get_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    def format_response(self, question: str, protein_context: Optional[Dict] = None, history: Optional[List[Dict]] = None) -> str:
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

        if protein_context is None:
            protein_context = {}

        if history is None:
            history = []

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

        messages = [{"role": "system", "content": system_prompt}]

        messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        return self.get_chat_completion(messages)