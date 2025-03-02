# Inf_ret_impl
real recognize real

## How to Use the Implementation

### Installation

First, install the required packages:

```bash
pip install PyPDF2 transformers torch
```

### Basic Usage

You can use the main script from the command line:

```bash
python main.py --document path/to/document.pdf --question "What is the main contribution of this research?" --model "gpt2" --verbose
```

Or for a text document:

```bash
python main.py --document path/to/document.txt --question "Who discovered penicillin?" --not-pdf --verbose
```

### Using a Larger Model

For better performance, you might want to use a larger model:

```bash
python main.py --document path/to/document.pdf --question "Summarize the key findings." --model "facebook/opt-1.3b" --verbose
```

### Tuning Parameters

You can experiment with different parameters:

```bash
python main.py --document path/to/document.pdf --question "What methodology was used?" --chunk-size 2048 --top-k 5 --phrase-len 4 --verbose
```

## Key Considerations and Improvements

1. **Memory Management**: When using larger models, you might need to manage memory carefully. Consider using quantization options like `-load-in-8bit` when initializing the model.
2. **Document Processing**: The current PDF extraction is basic. For complex PDFs with tables and multiple columns, consider using a more sophisticated library like `pdfplumber`.
3. **Caching Strategy**: You might want to develop more sophisticated caching strategies based on your specific use case, such as reranking or filtering cached sentences.
4. **Performance Optimization**: The current implementation prioritizes clarity over performance. For production use, you might want to optimize the code for speed and efficiency.