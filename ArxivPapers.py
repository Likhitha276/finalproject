import os
import time
import arxiv
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def arxiv_paper_serach(Topic, proficiency_level, content_depth, learning_context):
    search_query = f"{content_depth} of {Topic} for {proficiency_level} for {learning_context} purpose"
    max_results = 5
    dirpath="arxiv_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in client.results(search):
        st.write(f"Research papers found: {result}")
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred:", e)
                time.sleep(5)

    loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    papers = []
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
    
    full_text = ''
    for paper in papers:
        full_text += paper.page_content
    text_formatted = " ".join(line.strip() for line in full_text.splitlines() if line)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([text_formatted])

    vectorstore = Chroma.from_documents(documents=paper_chunks, embedding=OpenAIEmbeddings(openai_api_key = st.secrets["OPENAI_API_KEY"]))
    return vectorstore.as_retriever()

if __name__ == "__main__":
    topic = "Machine learning"
    proficiency_level = "Begineer"
    content_depth = "In depth"
    learning_context = "Academic"
    arxiv_paper_serach(Topic=topic, proficiency_level=proficiency_level, content_depth=content_depth,learning_context=learning_context)
