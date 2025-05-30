import torch
from torch.utils.data import Dataset
import gzip
from Bio import SeqIO
from datetime import datetime
import sys
import random
import pandas as pd
from tqdm import tqdm

class UniRefDataset(Dataset):
    def __init__(self, uniref_file, esm_model, alphabet, device, esm_layer, max_seq_len, seed_only=False, max_samples=50000000, random_seed=42, human_filter=1, return_difference=False):
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.max_seq_len = max_seq_len
        self.esm_layer = esm_layer
        self.seed_only = seed_only
        self.sequences = []
        self.metadata = []
        self.max_samples = max_samples
        self.go_ids = []
        self.go_terms = []
        self.batch_converter = alphabet.get_batch_converter()
        self.return_difference = return_difference

        # Set all random seeds for full reproducibility
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        print(f"Loading sequences from {uniref_file}...")
        print(f"Max sequence length: {max_seq_len}")
        print(f"Seed only: {seed_only}")
        print(f"Max samples to load: {max_samples:,}")
        print(f"Random seed: {random_seed}")
        print(f"Human filter: {'On' if human_filter else 'Off'}")
        sys.stdout.flush()

        print("\n=== Dataset Loading Process ===")
        print("1. Reading CSV file...")
        df = pd.read_csv(uniref_file, sep="\t")
        print(f"Found {len(df):,} total entries in the file")
        
        print("\n2. Processing sequences and collecting metadata...")
        # Initialize all eligible lists
        all_eligible_sequences = []
        all_eligible_metadata = []
        all_eligible_go_ids = []
        all_eligible_go_terms = []
        all_eligible_lengths = []
        all_eligible_go_biological = []
        all_eligible_go_cellular = []
        all_eligible_go_molecular = []
        all_eligible_entry_names = []
        all_eligible_protein_names = []
        all_eligible_gene_names = []
        all_eligible_organisms = []
        all_eligible_coiled_coil = []
        all_eligible_compositional_bias = []
        all_eligible_domain_cc = []
        all_eligible_domain_ft = []
        all_eligible_motif = []
        all_eligible_protein_families = []
        all_eligible_region = []
        all_eligible_repeat = []
        all_eligible_sequence_similarities = []
        all_eligible_zinc_finger = []
        
        skipped_sequences = 0
        skipped_non_human = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
            seq = row['Sequence']
            organism = row['Organism'] if pd.notna(row['Organism']) else None
            
            # Modified length and organism check based on human_filter
            length_ok = len(seq) <= max_seq_len
            organism_ok = organism == "Homo sapiens (Human)" if human_filter else True
            
            if length_ok and organism_ok:
                all_eligible_sequences.append(seq)
                all_eligible_metadata.append(row['Entry'])
                
                go_ids = str(row['Gene Ontology IDs']).split(';') if pd.notna(row['Gene Ontology IDs']) else []
                go_terms = str(row['Gene Ontology (GO)']).split(';') if pd.notna(row['Gene Ontology (GO)']) else []
                go_ids = [gid.strip() for gid in go_ids]
                go_terms = [term.strip() for term in go_terms]
                all_eligible_go_ids.append(go_ids)
                all_eligible_go_terms.append(go_terms)
                
                all_eligible_lengths.append(row['Length'] if pd.notna(row['Length']) else None)
                
                # Process GO term categories
                all_eligible_go_biological.append(
                    str(row['Gene Ontology (biological process)']).split(';') if pd.notna(row['Gene Ontology (biological process)']) else []
                )
                all_eligible_go_cellular.append(
                    str(row['Gene Ontology (cellular component)']).split(';') if pd.notna(row['Gene Ontology (cellular component)']) else []
                )
                all_eligible_go_molecular.append(
                    str(row['Gene Ontology (molecular function)']).split(';') if pd.notna(row['Gene Ontology (molecular function)']) else []
                )
                
                all_eligible_entry_names.append(row['Entry Name'] if pd.notna(row['Entry Name']) else None)
                all_eligible_protein_names.append(row['Protein names'] if pd.notna(row['Protein names']) else None)
                all_eligible_gene_names.append(row['Gene Names'] if pd.notna(row['Gene Names']) else None)
                all_eligible_organisms.append(row['Organism'] if pd.notna(row['Organism']) else None)
                
                all_eligible_coiled_coil.append(row['Coiled coil'] if pd.notna(row['Coiled coil']) else None)
                all_eligible_compositional_bias.append(row['Compositional bias'] if pd.notna(row['Compositional bias']) else None)
                all_eligible_domain_cc.append(row['Domain [CC]'] if pd.notna(row['Domain [CC]']) else None)
                all_eligible_domain_ft.append(row['Domain [FT]'] if pd.notna(row['Domain [FT]']) else None)
                all_eligible_motif.append(row['Motif'] if pd.notna(row['Motif']) else None)
                all_eligible_protein_families.append(row['Protein families'] if pd.notna(row['Protein families']) else None)
                all_eligible_region.append(row['Region'] if pd.notna(row['Region']) else None)
                all_eligible_repeat.append(row['Repeat'] if pd.notna(row['Repeat']) else None)
                all_eligible_sequence_similarities.append(row['Sequence similarities'] if pd.notna(row['Sequence similarities']) else None)
                all_eligible_zinc_finger.append(row['Zinc finger'] if pd.notna(row['Zinc finger']) else None)
            else:
                if len(seq) > max_seq_len:
                    skipped_sequences += 1
                if human_filter and organism != "Homo sapiens (Human)":
                    skipped_non_human += 1

        total_eligible = len(all_eligible_sequences)
        print(f"\nProcessing complete:")
        print(f"- Eligible sequences: {total_eligible:,}")
        print(f"- Skipped sequences (too long): {skipped_sequences:,}")
        print(f"- Skipped sequences (non-human): {skipped_non_human:,}")

        print("\n3. Sampling sequences...")
        # Randomly sample if we have more eligible sequences than max_samples
        if total_eligible > max_samples:
            print(f"Randomly sampling {max_samples:,} sequences from {total_eligible:,} eligible sequences...")
            indices = random.sample(range(total_eligible), max_samples)
            self.sequences = [all_eligible_sequences[i] for i in tqdm(indices, desc="Sampling")]
            self.metadata = [all_eligible_metadata[i] for i in indices]
            self.go_ids = [all_eligible_go_ids[i] for i in indices]
            self.go_terms = [all_eligible_go_terms[i] for i in indices]
            
            self.lengths = [all_eligible_lengths[i] for i in indices]
            self.go_biological = [all_eligible_go_biological[i] for i in indices]
            self.go_cellular = [all_eligible_go_cellular[i] for i in indices]
            self.go_molecular = [all_eligible_go_molecular[i] for i in indices]
            self.entry_names = [all_eligible_entry_names[i] for i in indices]
            self.protein_names = [all_eligible_protein_names[i] for i in indices]
            self.gene_names = [all_eligible_gene_names[i] for i in indices]
            self.organisms = [all_eligible_organisms[i] for i in indices]
            self.coiled_coil = [all_eligible_coiled_coil[i] for i in indices]
            self.compositional_bias = [all_eligible_compositional_bias[i] for i in indices]
            self.domain_cc = [all_eligible_domain_cc[i] for i in indices]
            self.domain_ft = [all_eligible_domain_ft[i] for i in indices]
            self.motif = [all_eligible_motif[i] for i in indices]
            self.protein_families = [all_eligible_protein_families[i] for i in indices]
            self.region = [all_eligible_region[i] for i in indices]
            self.repeat = [all_eligible_repeat[i] for i in indices]
            self.sequence_similarities = [all_eligible_sequence_similarities[i] for i in indices]
            self.zinc_finger = [all_eligible_zinc_finger[i] for i in indices]
        else:
            print("Using all eligible sequences (fewer than max_samples)...")
            self.sequences = all_eligible_sequences
            self.metadata = all_eligible_metadata
            self.go_ids = all_eligible_go_ids
            self.go_terms = all_eligible_go_terms
            
            self.lengths = all_eligible_lengths
            self.go_biological = all_eligible_go_biological
            self.go_cellular = all_eligible_go_cellular
            self.go_molecular = all_eligible_go_molecular
            self.entry_names = all_eligible_entry_names
            self.protein_names = all_eligible_protein_names
            self.gene_names = all_eligible_gene_names
            self.organisms = all_eligible_organisms
            self.coiled_coil = all_eligible_coiled_coil
            self.compositional_bias = all_eligible_compositional_bias
            self.domain_cc = all_eligible_domain_cc
            self.domain_ft = all_eligible_domain_ft
            self.motif = all_eligible_motif
            self.protein_families = all_eligible_protein_families
            self.region = all_eligible_region
            self.repeat = all_eligible_repeat
            self.sequence_similarities = all_eligible_sequence_similarities
            self.zinc_finger = all_eligible_zinc_finger

        print("\n=== Dataset Loading Summary ===")
        print(f"Final dataset size: {len(self.sequences):,} sequences")
        print(f"Average sequence length: {sum(len(seq) for seq in self.sequences)/len(self.sequences):.1f}")
        print("================================")
        sys.stdout.flush()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        metadata = self.metadata[idx]
        go_ids = self.go_ids[idx]
        go_terms = self.go_terms[idx]
        additional_metadata = {
            'length': self.lengths[idx],
            'go_biological': self.go_biological[idx],
            'go_cellular': self.go_cellular[idx],
            'go_molecular': self.go_molecular[idx],
            'entry_name': self.entry_names[idx],
            'protein_names': self.protein_names[idx],
            'gene_names': self.gene_names[idx],
            'organism': self.organisms[idx],
            'coiled_coil': self.coiled_coil[idx],
            'compositional_bias': self.compositional_bias[idx],
            'domain_cc': self.domain_cc[idx],
            'domain_ft': self.domain_ft[idx],
            'motif': self.motif[idx],
            'protein_families': self.protein_families[idx],
            'region': self.region[idx],
            'repeat': self.repeat[idx],
            'sequence_similarities': self.sequence_similarities[idx],
            'zinc_finger': self.zinc_finger[idx]
        }

        # Prepare data in the format expected by batch_converter
        data = [(metadata, seq)]
        
        # Convert using batch_converter
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        


        # Compute ESM embeddings for both current and next layer
        with torch.no_grad():
            results = self.esm_model(
                batch_tokens.to(self.device), 
                repr_layers=[self.esm_layer, self.esm_layer + 1],
                return_contacts=False
            )
        

        # Extract embeddings but no bos and eos tokens
        current_embedding = results["representations"][self.esm_layer][0, 1:-1]  # Shape: [L, D]
        next_embedding = results["representations"][self.esm_layer + 1][0, 1:-1]  # Shape: [L, D]
        
        # Mean pool both embeddings
        current_embedding = torch.mean(current_embedding, dim=0)  # Shape: [D]
        next_embedding = torch.mean(next_embedding, dim=0)  # Shape: [D]
        difference_embedding = next_embedding - current_embedding  # Shape: [D]
        
        if self.return_difference:
            return current_embedding, difference_embedding, metadata, seq, go_ids, go_terms, additional_metadata
        else:
            return current_embedding, next_embedding, metadata, seq, go_ids, go_terms, additional_metadata


