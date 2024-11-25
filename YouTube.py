import ast
import streamlit as st
from langchain_chroma import Chroma
from urllib.parse import urlparse, parse_qs
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def youTube_Search(Topic, proficiency_level, content_depth, learning_context):
    tool = YouTubeSearchTool()
    youTube_Search_prompt = f"{content_depth} of {Topic} for {proficiency_level} for {learning_context} purpose in english with transcript"
    videos = tool.run(f"{youTube_Search_prompt}, 5")
    text_formatted = ""

    for url in ast.literal_eval(videos):
        st.write(f"Video extracted from youTube: {url}")
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            formatter = TextFormatter()
            text_formatted += formatter.format_transcript(transcript)
        except Exception as e:
            print(f"Error: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([text_formatted])

    vectorstore = Chroma.from_documents(documents=paper_chunks, embedding=OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"]))
    return vectorstore.as_retriever()

if __name__ == '__main__':
    topic = "Machine learning"
    proficiency_level = "Begineer"
    content_depth = "In depth"
    learning_context = "Academic"
    youTube_Search(Topic=topic, proficiency_level=proficiency_level, content_depth=content_depth,learning_context=learning_context)