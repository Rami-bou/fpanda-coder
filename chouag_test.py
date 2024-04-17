from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.chat_models import ChatCohere, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from InstructorEmbedding import INSTRUCTOR


llm = ChatCohere()


st.set_page_config("Chouag test")
st.header("chat with BuddyBot")
user_query = st.text_input("Ask your question")

loader = UnstructuredFileLoader(r"C:\\Users\\rami\Downloads\\try.txt")
docs = loader.load()
chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size = 400,
    chunk_overlap = 0,
    length_function=len
)
chunks = text_splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

template = """
Answer the following questions based on the given document:

Document: {doc}

User question: {user_question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

if user_query: 
    v = vectorstore.similarity_search(query = user_query, k=3)
    answer = chain.invoke({
        'doc':v,
        'user_question':user_query
    })
    st.write(answer)
