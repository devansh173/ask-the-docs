from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def build_qa_chain(text):
    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedder)

    # Load light model (Flan-T5 Base)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Build QA Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    return chain