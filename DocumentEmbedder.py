import pymupdf
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import os
import re


class DocumentEmbedder:
    """
    A class for processing text documents (PDF, DOCX), including reading, chunking, embedding,
    and storing in a vector database.
    """

    def __init__(self, path, language='english'):
        """
        Initializes the DocumentEmbedder instance.
        This constructor can be used to initialize any required configurations
        or models for text processing.
        """
        self.path = path
        self.text = None

        self.language = language.lower()
        if self.language == 'english':
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.language == 'hebrew':
            self.embedding_model = SentenceTransformer('imvladikon/sentence-transformers-alephbert')
        else:
            raise ValueError("Unsupported language. Supported languages are English and Hebrew.")

        self.index = None
        self.chunk_mapping = None

    def embed_and_store(self, chunks):
        """
        Generates embeddings for the given text chunks using the current embedding model,
        and then stores these embeddings in the FAISS index along with a mapping of text chunks.
        Args:
            chunks (list[str]): A list of text chunks to be embedded and stored.
        Returns:
            None
        Raises:
            ValueError: If no text chunks are provided.
        """
        if not chunks:
            raise ValueError("No text chunks provided for embedding and storage.")

        embeddings = self.generate_embeddings(chunks)
        self.store_embeddings(embeddings, chunks)

    def store_embeddings(self, embeddings, chunks=None):
        """
        Stores embeddings in a FAISS vector database.

        Args:
            embeddings (np.ndarray): The embeddings to be stored.
            chunks (list, optional): The original text chunks corresponding to the embeddings.
                                     This mapping will be used during search to return the text.
        """
        if embeddings is None or embeddings.shape[0] == 0:
            raise ValueError("No embeddings to store.")

        # Save the mapping of chunks (if provided) for later retrieval.
        if chunks is not None:
            self.chunk_mapping = chunks

        # Determine the dimensionality (d) of the embeddings.
        d = embeddings.shape[1]
        # Create a FAISS index using the L2 distance metric.
        self.index = faiss.IndexFlatL2(d)
        # Add embeddings to the index.
        self.index.add(embeddings)
        print(f"Stored {embeddings.shape[0]} embeddings of dimension {d} in FAISS index.")

    def search_similar(self, query, k=5):
        """
        Searches for similar text based on embeddings using FAISS.
        Args:
            query (str): The query text.
            k (int, optional): Number of closest matches to return. Defaults to 5.
        Returns:
            list: A list of tuples (text_chunk, distance) for the closest matches.
        """
        if self.index is None:
            raise ValueError("Embeddings index is not built. Call store_embeddings() first.")

        # Generate an embedding for the query. It returns a 2D array with shape (1, d).
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        # Perform the search in the FAISS index.
        distances, indices = self.index.search(query_vector, k)
        results = []
        for j, idx in enumerate(indices[0]):
            # Retrieve the original text chunk if mapping is available.
            chunk_text = self.chunk_mapping[idx] if self.chunk_mapping is not None else None
            results.append((chunk_text, distances[0][j]))
        return results

    def generate_embeddings(self, chunks):
        """
        Generates embeddings for given text chunks.
        Args:
            chunks (list): A list of text chunks.
        Returns:
            np.ndarray: Array of embeddings.
        """
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        return embeddings


    def chunk_text(self, method="fixed", chunk_size=300, overlap=50, text=None):
        """
        Splits text into chunks using different methods.
        If text is not provided, it uses the instance attribute self.text.
        Args:
            method (str): Chunking method ("fixed", "sentence", "paragraph").
            chunk_size (int, optional): Size of fixed chunks. Defaults to 300.
            overlap (int, optional): Overlapping tokens/characters for fixed chunks. Defaults to 50.
            text (str, optional): The text to be split. If not provided, self.text is used.
        Returns:
            list: A list of text chunks.
        Raises:
            ValueError: If no text is provided.
        """
        if text is None:
            if self.text is None:
                raise ValueError("No text provided for chunking.")
            text = self.text

        if method == "fixed":
            chunk = self.fixed_chunking_with_overlap(text, chunk_size, overlap)
        elif method == "paragraph":
            chunk = self.paragraph_chunking(text)
        elif method == "line":
            chunk = self.line_chunking(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}. Use 'fixed, 'paragraph or 'line'")

        return self.adjust_chunk_order(chunk)

    def adjust_chunk_order(self, chunks):
        """
        Adjusts the order of chunks based on the language.
        For Hebrew ('hb'), reverse the list so that the first chunk corresponds to the
        beginning of the document in reading order.
        Args:
            chunks (list): The list of text chunks.
        Returns:
            list: The adjusted list of text chunks.
        """
        if self.language == "hebrew":
            print("succesfully checked for hebrew")
            return chunks[::-1]
        print("returned english")
        return chunks

    def paragraph_chunking(self, text: str = None) -> list[str]:
        """
        Splits the text into paragraphs using regex to detect two or more newlines.
        Cleans up unwanted newline characters within each paragraph.
        Args:
            text (str, optional): The text to be split. If not provided, self.text is used.
        Returns:
            list[str]: A list of clean paragraph chunks.
        """
        if text is None:
            if self.text is None:
                raise ValueError("No text provided for paragraph chunking.")
            text = self.text
        # Split text by multiple newlines
        paragraphs = re.split(r'\n\s*\n', text.strip())

        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove extra newlines within a paragraph
            para = re.sub(r'\s*\n\s*', ' ', para).strip()

            # Remove isolated numbers, single characters, and overly short paragraphs
            if len(para) > 5 and not para.isdigit():
                cleaned_paragraphs.append(para)

        return cleaned_paragraphs

    def line_chunking(self, text: str = None) -> list[str]:
        """
        Splits the provided text into chunks based on individual lines.
        Each non-empty line becomes a separate chunk.
        Args:
            text (str, optional): The text to be split.
        Returns:
            list[str]: A list of line chunks.
        """
        if text is None:
            raise ValueError("No text provided for line chunking.")
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
        return chunks

    def fixed_chunking_with_overlap(self, text: str = None, chunk_size: int = 300, overlap: int = 50) -> list[str]:
        """
        Splits the text into fixed-size chunks based on tokens, with a specified overlap.
        Args:
            text (str, optional): The text to be split. If not provided, self.text is used.
            chunk_size (int): The number of tokens in each chunk.
            overlap (int): The number of tokens to overlap between consecutive chunks.
        Returns:
            list[str]: A list of text chunks.

        Raises:
            ValueError: If overlap is greater than or equal to chunk_size.
        """
        if text is None:
            if self.text is None:
                raise ValueError("No text provided for fixed chunking.")
            text = self.text

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than the chunk size.")

        tokens = text.split()
        total_tokens = len(tokens)
        chunks = []
        i = 0

        while i < total_tokens:
            chunk_tokens = tokens[i: i + chunk_size]
            chunk = " ".join(chunk_tokens)
            chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def extract_text(self, file_path):
        """
        Determines the file type (PDF or DOCX) and extracts text accordingly.
        Args:
            file_path (str): The path of the file.
        Returns:
            str: Extracted text.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            self.text = self.extract_text_from_pdf(file_path)
            return self.extract_text_from_pdf(file_path)
        elif file_extension == ".docx":
            self.text = self.extract_text_from_docx(file_path)
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a given PDF file.
        Args:
            pdf_path (str): The path to the PDF file.
        Returns:
            str: The extracted text.
        """
        try:
            doc = pymupdf.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
            return text
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {pdf_path}")
        except Exception as e:
            raise ValueError(f"Error processing PDF file: {str(e)}")

    def extract_text_from_docx(self, docx_path):
        """
        Extracts text from a given DOCX file.
        Args:
            docx_path (str): The path to the DOCX file.
        Returns:
            str: The extracted text.
        """
        try:
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {docx_path}")
        except Exception as e:
            raise ValueError(f"Error processing DOCX file: {str(e)}")
