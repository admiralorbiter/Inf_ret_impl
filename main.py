import argparse
import time
import torch
from chunking import DocumentChunker
from attention_retrieval import AttentionRetriever
from cache_manager import CacheManager

def answer_question_with_infinitretri(
    document_path,
    question,
    model_name="gpt2",
    chunk_size=1024,
    top_k=3,
    phrase_len=3,
    is_pdf=True,
    max_cache_size=None,
    verbose=False
):
    """
    Answer a question about a document using the InfiniRetri attention-based retrieval system.
    
    Args:
        document_path (str): Path to the document
        question (str): Question to answer
        model_name (str): HuggingFace model name
        chunk_size (int): Maximum token size for each chunk
        top_k (int): Number of top attention indices to consider per chunk
        phrase_len (int): Size of phrase window for attention analysis
        is_pdf (bool): Whether the document is a PDF
        max_cache_size (int): Maximum number of sentences to cache
        verbose (bool): Whether to print debug information
        
    Returns:
        tuple: (answer, cached_context)
    """
    start_time = time.time()
    
    # Initialize components
    chunker = DocumentChunker(tokenizer_name=model_name, chunk_size_tokens=chunk_size)
    retriever = AttentionRetriever(model_name=model_name, phrase_len=phrase_len)
    cache_mgr = CacheManager(max_cache_size=max_cache_size)
    
    # Extract and chunk the document
    if verbose:
        print(f"Processing document: {document_path}")
    chunks = chunker.process_document(document_path, is_pdf=is_pdf)
    if verbose:
        print(f"Document split into {len(chunks)} chunks")
    
    # Process each chunk and update cache
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        
        # Merge current chunk with cached text
        merged_text = cache_mgr.get_merged_text(chunk)
        
        # Get attention scores for tokens in this merged text
        importance_scores, offset_mapping, tokens = retriever.get_attention_scores(merged_text, question)
        
        # Get top-k token indices based on importance scores
        top_indices = retriever.get_top_k_indices(importance_scores, k=top_k)
        
        # Map token indices to character positions
        top_char_positions = [offset_mapping[idx] for idx in top_indices]
        
        # Extract sentences containing these positions from the chunk
        important_sentences = cache_mgr.extract_sentences_from_text(chunk, top_char_positions)
        
        # Add sentences to cache
        for sentence in important_sentences:
            cache_mgr.add_sentence(sentence)
            
        if verbose:
            print(f"  Found {len(important_sentences)} important sentences, cache now has {len(cache_mgr)} sentences")
    
    # Generate final answer using the cached context
    cached_context = cache_mgr.get_cached_text()
    
    # Prepare the final prompt for answer generation
    final_prompt = f"Context information:\n{cached_context}\n\nBased on the above context only, answer the following question once:\nQuestion: {question}\nAnswer:"
    
    # Generate the answer
    input_ids = retriever.tokenizer.encode(final_prompt, return_tensors='pt').to(retriever.device)
    with torch.no_grad():
        output_ids = retriever.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 200,  # Limit answer length
            pad_token_id=retriever.tokenizer.pad_token_id,
            num_return_sequences=1
        )
    
    # Decode only the newly generated tokens (the answer)
    answer = retriever.tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    end_time = time.time()
    if verbose:
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Final context size: {len(cached_context)} chars")
    
    return answer, cached_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniRetri: Attention-based document retrieval system")
    parser.add_argument("--document", type=str, required=True, help="Path to document file")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in tokens")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top attention indices to consider")
    parser.add_argument("--phrase-len", type=int, default=3, help="Size of phrase window for attention")
    parser.add_argument("--not-pdf", action="store_true", help="Document is not a PDF")
    parser.add_argument("--max-cache", type=int, default=None, help="Maximum cache size (sentences)")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    answer, context = answer_question_with_infinitretri(
        document_path=args.document,
        question=args.question,
        model_name=args.model,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        phrase_len=args.phrase_len,
        is_pdf=not args.not_pdf,
        max_cache_size=args.max_cache,
        verbose=args.verbose
    )
    
    # Filter the answer to only include the first sentence or up to the first question
    if "\nQuestion:" in answer:
        answer = answer.split("\nQuestion:")[0].strip()
    elif "." in answer:
        answer = answer.split(".")[0].strip() + "."
    
    print("\n" + "="*50)
    print(f"Question: {args.question}")
    print("="*50)
    print(f"Answer: {answer}")
    print("="*50)
    print(f"Retrieved Context:")
    print("-"*50)
    print(context)
    print("="*50)
