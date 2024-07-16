import os
import streamlit as st
import pickle
import time
from google.cloud import aiplatform
# from langchain import OpenAIpip
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import  UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA



from dotenv import load_dotenv
aiplatform.init(
    project="My Project 71083",
    location="us-central1",
)

st.title("News Researcher")
st.sidebar.title("URLS to extract")
urls=[]
filepath="faiss_store_openai.pkl"
llm=GooglePalm(temperature=0.9,google_api_key="")
for i  in range(3):
    urls.append(st.sidebar.text_input(f"URL{i+1}"))


process_url_clicked = st.sidebar.button("Extract URLS")

main_placefolder=st.empty()
if process_url_clicked :
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading Started")
    data=loader.load()

    text_splitter= RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," "],
        chunk_size=1000
    )
    main_placefolder.text("Splitting Text Started")

    docs=text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    main_placefolder.text("Embedded Vectorstore Started Building")

    with open(filepath,"wb") as f :
        pickle.dump(vectorstore_openai,f)

query = main_placefolder.text_input("Question")

if query:
    if os.path.exists(filepath):

        with open(filepath,"rb") as f :
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",
                return_source_documents=True)
            result=chain(query)
            st.header("Answer")
            st.subheader(result["result"])

            sources=result.get("sources","")
            if sources:
                st.subheader("Sources")
                sources_list=sources.split("\n")
                if sources_list:  # Check if the list is not empty before iterating
                    for source in sources_list:
                        st.write(source)


