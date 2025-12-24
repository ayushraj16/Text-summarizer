from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


def generate_summary(text):

    # 1. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # 2. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Store in FAISS
    vector_db = FAISS.from_texts(chunks, embeddings)

    # 4. Retrieve most relevant chunks
    docs = vector_db.similarity_search(
        "Summarize the document",
        k=4
    )

    context = " ".join([doc.page_content for doc in docs])

    # 5. Load summarization model (CPU)
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200,
        device=-1
    )

    # 6. Generate summary
    summary = summarizer(
        f"Summarize the following content:\n{context}"
    )[0]["generated_text"]

    return summary
