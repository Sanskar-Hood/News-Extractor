import os
import streamlit as st
import pickle
import time
from google.cloud import aiplatform
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import  UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA


from dotenv import load_dotenv
aiplatform.init(
    project="My Project 71083",
    location="us-central1",
)
#
# st.title("News Researcher")
# st.sidebar.title("URLS to extract")
urls=[]
filepath="faiss_store_openai.pkl"
llm=GooglePalm(temperature=0.9,google_api_key="AIzaSyBRc70rVG-cVdtPlz8_otN3MgE8396ZH_c")
# for i  in range(3):
#     urls.append(st.sidebar.text_input(f"URL{i+1}"))
#
#
# process_url_clicked = st.sidebar.button("Extract URLS")
#
# main_placefolder=st.empty()

loader=UnstructuredURLLoader(urls=["https://indianexpress.com/article/sports/cricket/what-are-five-unique-stats-of-james-andersons-test-career-9449595/"])

data=loader.load()
# print(data)

text_splitter= RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," "],
        chunk_size=1000
    )


docs=text_splitter.split_documents(data)
# docs=text_splitter.split_text(dat

#
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# print(embeddings.embed_documents(["sagdyasgdyjgasjdyhgasjhdgasjhdgjhasgdjhsvd"]))
#
vectorstore_openai = FAISS.from_documents(docs,embeddings)

#
#
with open(filepath,"wb") as f :
    pickle.dump(vectorstore_openai,f)
#
query = "what age anderson retired?"
#
if query:
    if os.path.exists(filepath):

        with open(filepath,"rb") as f :
            vectorstore = pickle.load(f)
            # chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            retriever = vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",
                return_source_documents=True)

            result=chain(query)
            print(result)
