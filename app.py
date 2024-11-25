__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from RAG import RAG_chat

st.title("Customized E-Learning")

with st.sidebar:
    st.header("Customize Your Learning")
    topic = st.text_input("Enter the specific skill or topic you want to learn:")
    proficiency_level = st.selectbox("Proficiency Level", ["Beginner", "Intermediate", "Advance"])
    content_depth = st.selectbox("Content Depth", ["High-level overview", "In-Depth"])
    learning_platform = st.selectbox("Learning Platforms", ["YouTube", "ArXiv papers", "Wikipedia"])
    learning_context = st.selectbox("Learning Context", ["Academic", "Professional Development", "Personal Interest"])
    llm = RAG_chat(topic=topic, proficiency_level=proficiency_level, content_depth=content_depth, learning_platform=learning_platform, learning_context=learning_context)
    if st.button("Submit"):
        st.session_state["messages"] = []

        st.session_state["submitted_options"] = {
            "topic": topic,
            "proficiency_level": proficiency_level,
            "content_depth": content_depth,
            "learning_platform": learning_platform,
            "learning_context": learning_context
        }
        st.success("Learning preferences submitted")

st.subheader("AI Learning Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Ask your question about the topic:")

if user_input and st.session_state.get("submitted_options"):
    response = llm.chat(user_input)
    st.session_state["messages"].append({"user": user_input, "bot": response})

user_message_style = """
    <div style='background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin-bottom: 10px;
    font-size: 16px; font-weight: bold; color: #000;'>
        <strong>You:</strong> {}
    </div>
"""
bot_message_style = """
    <div style='background-color: #f8d7da; padding: 10px; border-radius: 10px; margin-bottom: 10px;
    font-size: 16px; font-weight: bold; color: #000;'>
        <strong>AI:</strong> {}
    </div>
"""

for message in st.session_state["messages"]:
    st.markdown(user_message_style.format(message['user']), unsafe_allow_html=True)
    st.markdown(bot_message_style.format(message['bot']), unsafe_allow_html=True)
    
