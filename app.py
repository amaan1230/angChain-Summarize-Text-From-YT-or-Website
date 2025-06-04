import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Optional fallback loader
try:
    from langchain_community.document_loaders.youtube import YouTubeTranscriptLoader

    def load_youtube_transcript(url):
        loader = YouTubeTranscriptLoader(url)
        docs = loader.load()
        return docs

except ImportError:
    from youtube_transcript_api import YouTubeTranscriptApi
    import re
    from langchain_core.documents import Document

    def extract_video_id(url):
        regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(regex, url)
        return match.group(1) if match else url

    def load_youtube_transcript(url):
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t['text'] for t in transcript])
        return [Document(page_content=text)]

# Streamlit App Config
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL (YouTube or Website)")

# Sidebar Inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# URL Input
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="visible")

# Groq LLM (Llama3 is available)
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in about 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Main Action
if st.button("Summarize the Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the API key and the URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url])
                    docs = loader.load()

                # Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success("Summary generated successfully!")
                st.write(summary)
        except Exception as e:
            st.error("❌ Error occurred. Please check the link or API key.")
            st.exception(e)
