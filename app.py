import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_pipeline import generate_summary


st.set_page_config(page_title="PDF RAG Summarizer", layout="centered")

st.title("📄 PDF Summarizer using RAG")
st.write("Upload a PDF and get an intelligent summary using Retrieval-Augmented Generation.")


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if len(pdf_text) < 200:
        st.warning("PDF content too short for summarization.")
    else:
        with st.spinner("Building RAG pipeline..."):
            rag_chain = generate_summary(pdf_text)

        if st.button("Generate Summary"):
          with st.spinner("Generating summary..."):
           summary = generate_summary(pdf_text)

          st.subheader("📌 Summary")
          st.success(summary)

