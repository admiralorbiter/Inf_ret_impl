import re

class CacheManager:
    def __init__(self, max_cache_size=None):
        """
        Initialize the cache manager for storing relevant sentences.
        
        Args:
            max_cache_size (int, optional): Maximum number of sentences to store in cache
        """
        self.sentences = []  # List of cached sentences
        self.max_cache_size = max_cache_size
    
    def add_sentence(self, sentence):
        """
        Add a sentence to the cache if it's not already present.
        
        Args:
            sentence (str): Sentence to add to cache
            
        Returns:
            bool: True if sentence was added, False if it was already in cache
        """
        # Normalize sentence (trim whitespace, etc.)
        sentence = sentence.strip()
        
        # Skip empty sentences
        if not sentence:
            return False
            
        # Check if sentence is already in cache
        if sentence in self.sentences:
            return False
            
        # Add the sentence
        self.sentences.append(sentence)
        
        # If we've exceeded max cache size, remove oldest sentence
        if self.max_cache_size and len(self.sentences) > self.max_cache_size:
            self.sentences.pop(0)
            
        return True
    
    def get_cached_text(self):
        """
        Get the combined text of all cached sentences.
        
        Returns:
            str: Combined cached text
        """
        return " ".join(self.sentences)
    
    def get_merged_text(self, new_chunk_text):
        """
        Merge cached sentences with new chunk text.
        
        Args:
            new_chunk_text (str): New chunk text to append
            
        Returns:
            str: Combined text (cache + new chunk)
        """
        cache_text = self.get_cached_text()
        
        if cache_text:
            return cache_text + " " + new_chunk_text
        return new_chunk_text
    
    def extract_sentences_from_text(self, text, char_positions):
        """
        Extract full sentences from text given character positions of important tokens.
        
        Args:
            text (str): The text to extract sentences from
            char_positions (list): List of (start, end) character positions
            
        Returns:
            list: List of extracted sentences
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Keep track of character offset for each sentence
        char_offset = 0
        sentence_ranges = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Calculate start and end positions for this sentence
            sentence_len = len(sentence)
            start_pos = char_offset
            # Add 1 for the space after the sentence (or punctuation)
            end_pos = char_offset + sentence_len + 1
            
            sentence_ranges.append((start_pos, end_pos, sentence))
            char_offset = end_pos
        
        # Find which sentences contain our important character positions
        extracted_sentences = set()
        
        for start_pos, end_pos in char_positions:
            for sent_start, sent_end, sentence in sentence_ranges:
                # If the character position falls within this sentence's range
                if (start_pos >= sent_start and start_pos < sent_end) or \
                   (end_pos > sent_start and end_pos <= sent_end):
                    extracted_sentences.add(sentence)
                    break
        
        return list(extracted_sentences)
    
    def clear_cache(self):
        """
        Clear all cached sentences.
        """
        self.sentences = []
    
    def __len__(self):
        """
        Get the number of sentences in the cache.
        
        Returns:
            int: Number of cached sentences
        """
        return len(self.sentences)
