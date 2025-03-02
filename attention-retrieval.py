import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class AttentionRetriever:
    def __init__(self, model_name="gpt2", phrase_len=3, device=None):
        """
        Initialize the attention-based retriever.
        
        Args:
            model_name (str): HuggingFace model name
            phrase_len (int): Size of the sliding window for phrase-level attention
            device (str): Device to run the model on (None for auto-detection)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine device (CPU/GPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,  # Important for attention retrieval
            device_map=self.device
        )
        self.model.eval()  # Set to evaluation mode
        
        self.phrase_len = phrase_len
        print("Retriever ready.")
    
    def get_attention_scores(self, input_text, question_text):
        """
        Get attention-based importance scores for each token in the input context.
        
        Args:
            input_text (str): The input context (cache + current chunk)
            question_text (str): The question/query text
            
        Returns:
            tuple: (importance_scores, token_to_char_map, all_tokens)
              - importance_scores (list): List of importance scores for each context token
              - token_to_char_map (list): Mapping from token indices to character positions
              - all_tokens (list): All tokenized input for reference
        """
        # Prepare input: context + question
        full_input = input_text + " Question: " + question_text
        
        # Tokenize the input, tracking original positions
        encoding = self.tokenizer(full_input, return_tensors='pt', return_offsets_mapping=True)
        input_ids = encoding["input_ids"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()  # Character positions for each token
        
        # Find where the question starts
        question_start_str = " Question: "
        question_start_pos = full_input.find(question_start_str) + len(question_start_str)
        
        # Find which token corresponds to the start of the question
        context_token_end_idx = 0
        for i, (start, end) in enumerate(offset_mapping):
            if start >= question_start_pos:
                context_token_end_idx = i
                break
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        # Get the last layer attention
        # Shape: [batch, num_heads, seq_len, seq_len]
        last_attn = outputs.attentions[-1][0]  # Get first batch
        
        # Sum over all attention heads
        attn_matrix = last_attn.sum(dim=0).to("cpu")  # Shape: [seq_len, seq_len]
        
        # Define the indices for context and question tokens
        seq_len = attn_matrix.shape[0]
        context_indices = list(range(0, context_token_end_idx))
        question_indices = list(range(context_token_end_idx, seq_len))
        
        # Calculate importance scores using convolution over attention values
        # For each context position, we'll compute a "phrase attention score"
        importance_scores = [0.0] * len(context_indices)
        
        # For each question token
        for q_idx in question_indices:
            # Get this question token's attention to all context tokens
            q_attn = attn_matrix[q_idx, :context_token_end_idx]
            
            # Use a convolution to get phrase-level attention
            # We'll use a sliding window of size self.phrase_len
            if len(context_indices) >= self.phrase_len:
                # Use actual convolution for efficiency
                # Reshape to [1, 1, len] for 1D convolution
                q_attn_reshaped = q_attn.view(1, 1, -1)
                # Create a filter of ones with size phrase_len
                conv_filter = torch.ones(1, 1, self.phrase_len)
                # Apply convolution (with proper padding)
                padding = self.phrase_len - 1
                phrase_attn = F.conv1d(
                    q_attn_reshaped, 
                    conv_filter, 
                    padding=padding
                )
                # Remove the padding at the end
                phrase_attn = phrase_attn[0, 0, :len(context_indices)]
                
                # Add to importance scores
                for i, score in enumerate(phrase_attn.tolist()):
                    importance_scores[i] += score
            else:
                # If context is smaller than phrase_len, just use raw attention
                for i, score in enumerate(q_attn.tolist()):
                    importance_scores[i] += score
        
        # Get all tokens for reference and debugging
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return importance_scores, offset_mapping[:context_token_end_idx], all_tokens[:context_token_end_idx]
    
    def get_top_k_indices(self, importance_scores, k=3):
        """
        Get the indices of the top-k highest scoring tokens.
        
        Args:
            importance_scores (list): List of importance scores
            k (int): Number of top indices to return
            
        Returns:
            list: Indices of the top-k highest scoring tokens
        """
        # Sort indices by score (highest first)
        sorted_indices = sorted(
            range(len(importance_scores)), 
            key=lambda i: importance_scores[i], 
            reverse=True
        )
        
        # Return top-k indices
        return sorted_indices[:k]
