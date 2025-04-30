import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional
import re
import requests
from Bio import SeqIO
from io import StringIO

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
    
    def extract_protein_sequence(self, text: str) -> Optional[str]:
        """
        Extract protein sequence from text using regex
        """
        # Pattern for protein sequences (uppercase letters, possibly with spaces)
        pattern = r'[A-Z\s]{10,}'
        matches = re.findall(pattern, text)
        
        if matches:
            # Clean up the sequence (remove spaces, etc.)
            sequence = ''.join(matches[0].split())
            # Basic validation (should only contain valid amino acids)
            if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
                return sequence
        return None
    
    def get_protein_from_uniprot(self, protein_name: str) -> Optional[Dict]:
        """
        Fetch protein information from UniProt
        """
        try:
            # Search UniProt
            search_url = f"https://www.uniprot.org/uniprot/?query={protein_name}&format=fasta"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                # Parse FASTA
                fasta = StringIO(response.text)
                record = next(SeqIO.parse(fasta, "fasta"))
                
                return {
                    "protein_name": record.id,
                    "sequence": str(record.seq),
                    "description": record.description
                }
        except Exception as e:
            print(f"Error fetching from UniProt: {str(e)}")
        return None
    
    def get_protein_context(self, question: str) -> Dict:
        """
        Extract protein-related context from the question
        """
        context = {
            "protein_name": None,
            "sequence": None,
            "embeddings": None,
            "description": None
        }
        
        # Try to extract protein sequence directly from question
        sequence = self.extract_protein_sequence(question)
        if sequence:
            context["sequence"] = sequence
            context["embeddings"] = self.get_protein_embeddings(sequence)
            return context
        
        # If no sequence found, look for protein names
        # This is a simple implementation - could be improved with NER
        protein_keywords = ['protein', 'enzyme', 'peptide', 'amino acid']
        if any(keyword in question.lower() for keyword in protein_keywords):
            # Extract potential protein names (simple implementation)
            words = question.split()
            for word in words:
                if word.isupper() and len(word) > 2:  # Simple heuristic for protein names
                    uniprot_data = self.get_protein_from_uniprot(word)
                    if uniprot_data:
                        context.update(uniprot_data)
                        context["embeddings"] = self.get_protein_embeddings(uniprot_data["sequence"])
                        return context
        
        return context 