from DocumentEmbedder import DocumentEmbedder

import os


def test_file(file_path, language):
    print("============================================")
    print(f"Processing file: {file_path} (Language: {language})")

    # Create an instance of DocumentEmbedder for the given file and language
    embedder = DocumentEmbedder(path=file_path, language=language)

    # Extract text from the file
    try:
        text = embedder.extract_text(file_path)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return


    # Chunk the text using a method of your choice (here we use paragraph chunking)
    try:
        chunks = embedder.chunk_text(method="paragraph")
    except Exception as e:
        print(f"Error during chunking: {e}")
        return

    print("\nChunks:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {chunk}")

    # Generate embeddings for the chunks and store them in the FAISS index
    try:
        embedder.embed_and_store(chunks)
    except Exception as e:
        print(f"Error during embedding and storage: {e}")
        return

    # Define a sample query for similarity search.
    # Use a query in English for English files and in Hebrew for Hebrew files.
    query = "test query" if language.lower() == "english" else "בדיקה"

    # Perform a similarity search (retrieve the 3 closest chunks)
    try:
        results = embedder.search_similar(query, k=3)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return

    print("\nSearch Results:")
    for result, distance in results:
        print(f"Chunk: {result}\nDistance: {distance}\n")
    print("============================================\n")




if __name__ == "__main__":
    test_file('test_file_hb.docx', 'english')
