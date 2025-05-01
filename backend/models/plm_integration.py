import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer
import numpy as np
from typing import Dict, List, Optional
import re
import requests
from Bio import SeqIO
from io import StringIO

class ProteinModel:
    def __init__(self):
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

        self.prot_t5_tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            legacy=True,
            model_max_length=512
        )
        self.prot_t5_model = AutoModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

        self.esm_model.eval()
        self.prot_t5_model.eval()

    def get_protein_embeddings(self, sequence: str, model_type: str = "esm") -> np.ndarray:
        if model_type == "esm":
            tokenizer = self.esm_tokenizer
            model = self.esm_model
        else:
            tokenizer = self.prot_t5_tokenizer
            model = self.prot_t5_model

        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        return embeddings

    def extract_protein_sequence(self, text: str) -> Optional[str]:
        pattern = r'[A-Z\s]{10,}'
        matches = re.findall(pattern, text)

        if matches:
            sequence = ''.join(matches[0].split())
            if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
                return sequence
        return None

    def get_protein_from_uniprot(self, protein_name: str) -> Optional[Dict]:
        try:
            search_url = f"https://www.uniprot.org/uniprot/?query={protein_name}&format=fasta"
            response = requests.get(search_url)

            if response.status_code == 200:
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
        context = {
            "protein_name": None,
            "sequence": None,
            "embeddings": None,
            "description": None
        }

        sequence = self.extract_protein_sequence(question)
        if sequence:
            context["sequence"] = sequence
            context["embeddings"] = self.get_protein_embeddings(sequence)
            return context

        protein_keywords = ['protein', 'enzyme', 'peptide', 'amino acid']
        if any(keyword in question.lower() for keyword in protein_keywords):
            words = question.split()
            for word in words:
                if word.isupper() and len(word) > 2:
                    uniprot_data = self.get_protein_from_uniprot(word)
                    if uniprot_data:
                        context.update(uniprot_data)
                        context["embeddings"] = self.get_protein_embeddings(uniprot_data["sequence"])
                        return context

        return context