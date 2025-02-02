# Document Embedder 📄🔍

A python based document processing tool that extracts text from PDF and DOCX files, chunks the text using multiple strategies, generates semantic embeddings and stores them using FFAISS.

## Features
Extracts Texts - supports PDF and DOCX  
Chunking - Fixed size with overlap, sentence based and paragraph based  
Multi Language Support - English and Hebrew  
Embeddings Generation - Uses SentenceTransformers models  
Efficient Vector Search - Stores and retrevies embeddings using FAISS  

### Installation  
```bash
git clone https://github.com/dan4151/document-embedder.git  
cd document-embedder
```

### Install dependencies:  
```bash
 pip install --upgrade pymupdf  
 pip install python-docx  
 pip install faiss-cpu  
 pip install tf-keras  
 pip install -U sentence-transformers
```

### Basic Example
```python
from document_embedder import DocumentEmbedder

# Initialize the embedder
embedder = DocumentEmbedder("test_file.pdf", language="english")

# Extract text from document
text = embedder.extract_text(embedder.path)

# Chunk the text
chunks = embedder.chunk_text(method="fixed", chunk_size=300, overlap=20)

# Generate and store embeddings
embedder.embed_and_store(chunks)

# Search for similar text
results = embedder.search_similar("Your query text", k=3)

# Print results
for chunk, distance in results:
    print(f"Chunk: {chunk}\nDistance: {distance}\n")
```

Once you have chunked and embedded your document, the chunks and their corresponding vector representations are stored in FAISS.

```python
# Check stored chunks
print("Stored Chunks:", embedder.chunk_mapping)

# Check FAISS index status
print("Index Size:", embedder.index.ntotal)  # Number of stored embeddings

# Search for a similar sentence
query = "What is the conclusion of the study?"
results = embedder.search_similar(query, k=3)

# Print results
for text, distance in results:
    print(f"Similar Chunk: {text} \nDistance: {distance}\n")
```




  
 
