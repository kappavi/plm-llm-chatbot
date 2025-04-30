import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional

class ProteinModel:
    def __init__(self):
        # Initialize ESM model
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        # Initialize ProtT5 model with specific tokenizer configuration
        self.prot_t5_tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            use_fast=False,  # Use slow tokenizer instead of fast tokenizer
            legacy=True,     # Use legacy tokenizer
            model_max_length=512  # Set maximum sequence length
        )
        self.prot_t5_model = AutoModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        
        # Set models to evaluation mode
        self.esm_model.eval()
        self.prot_t5_model.eval()
        
    def get_protein_embeddings(self, sequence: str, model_type: str = "esm") -> np.ndarray:
        """
        Get protein embeddings from either ESM or ProtT5 model
        """
        if model_type == "esm":
            tokenizer = self.esm_tokenizer
            model = self.esm_model
        else:
            tokenizer = self.prot_t5_tokenizer
            model = self.prot_t5_model
            
        # Tokenize sequence
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            
        return embeddings
    
    def get_protein_context(self, question: str) -> Dict:
        """
        Extract protein-related context from the question
        """
        # TODO: Implement protein sequence extraction from question
        # This could involve:
        # 1. Named entity recognition for protein names
        # 2. Sequence extraction from databases
        # 3. Structure prediction if needed
        
        # For now, return a placeholder
        return {
            "protein_name": "example_protein",
            "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "embeddings": self.get_protein_embeddings("MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
        } 