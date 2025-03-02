import re
from PyPDF2 import PdfReader
from transformers import AutoTokenizer

class DocumentChunker:
    def __init__(self, tokenizer_name="gpt2", chunk_size_tokens=1024):
        """
        Initialize the document chunker with a tokenizer and chunk size.
        
        Args:
            tokenizer_name (str): HuggingFace tokenizer name to use for token counting
            chunk_size_tokens (int): Maximum number of tokens per chunk
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size_tokens = chunk_size_tokens
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        reader = PdfReader(pdf_path)
        text_pages = [page.extract_text() for page in reader.pages]
        text = "\n".join(text_pages)
        return text
    
    def extract_text_from_file(self, file_path):
        """
        Extract text from a text file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: Text content
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
            
    def count_tokens(self, text):
        """
        Count the number of tokens in a text.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))
            
    def split_text_to_chunks(self, text):
        """
        Split text into chunks of approximately chunk_size_tokens.
        Respects sentence boundaries to maintain coherence.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of text chunks
        """
        # Split text into sentences using common punctuation as separators
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Count tokens in this sentence
            sentence_token_count = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content,
            # finalize the current chunk and start a new one
            if current_token_count + sentence_token_count > self.chunk_size_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            
            # If a single sentence is too long, we'll have to split it
            # (this is a simplified approach - in practice you might want more sophisticated sentence splitting)
            if sentence_token_count > self.chunk_size_tokens:
                # For simplicity, we'll just add it to its own chunk
                # A more sophisticated approach would split by clauses or add truncation logic
                chunks.append(sentence)
                continue
                
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        # Add the last chunk if it contains anything
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def process_document(self, input_path, is_pdf=True):
        """
        Process a document file and split it into chunks.
        
        Args:
            input_path (str): Path to the document
            is_pdf (bool): Whether the document is a PDF
            
        Returns:
            list: List of text chunks
        """
        if is_pdf:
            text = self.extract_text_from_pdf(input_path)
        else:
            text = self.extract_text_from_file(input_path)
            
        chunks = self.split_text_to_chunks(text)
        return chunks
