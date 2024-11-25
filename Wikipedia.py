import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter

def Wikipedia_search(Topic, proficiency_level, content_depth, learning_context):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    query = f"{content_depth} of {Topic} for {proficiency_level} for {learning_context} purpose"
    search_results = wikipedia.run(query)
    st.write(f"Extracting the wiki data for {query}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([search_results])

    vectorstore = Chroma.from_documents(documents=paper_chunks, embedding=OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"]))
    return vectorstore.as_retriever()

if __name__ == "__main__":
    topic = "Machine learning"
    proficiency_level = "Begineer"
    content_depth = "In depth"
    learning_context = "Academic"
    Wikipedia_search(Topic=topic, proficiency_level=proficiency_level, content_depth=content_depth,learning_context=learning_context)
