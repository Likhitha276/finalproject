from YouTube import youTube_Search
from Wikipedia import Wikipedia_search
from langchain_openai import ChatOpenAI
from ArxivPapers import arxiv_paper_serach
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAG_chat():
    def __init__(self, topic, proficiency_level,content_depth,learning_platform,learning_context) -> None:
        if learning_platform == "YouTube":
            self.retriever = youTube_Search(topic, proficiency_level, content_depth, learning_context)
        elif learning_platform == "ArXiv papers":
            self.retriever = arxiv_paper_serach(topic, proficiency_level, content_depth, learning_context)
        elif learning_platform == "Wikipedia":
            self.retriever = Wikipedia_search(topic, proficiency_level, content_depth, learning_context)

    def chat(self, prompt):  
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        llm = ChatOpenAI(model="gpt-4o-mini")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(prompt)

