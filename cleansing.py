# === cleansing.py ===
def cleansing(path):
    import os
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as read:
        wholeText = read.read()

    QApart = wholeText.split("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if len(QApart) < 2:
        print("[ERROR] The expected delimiter was not found in the file.")
        return []

    Flag = False
    CleanText = ""
    for e in QApart[1].split("\n"):
        if "?" in e or Flag:
            CleanText += e + "\n"
            Flag = False
        if "?" in e:
            Flag = True

    return CleanText.strip().split("\n")


# === rag.py ===
import os
import gc
import json
import torch
import chromadb
import streamlit as st
from gensim.parsing.preprocessing import preprocess_documents
from chromadb.utils import embedding_functions
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

llm = CTransformers(
    model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    model_file='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    config={'max_new_tokens': 4096, 'temperature': 0.2, 'context_length': 8000},
    threads=os.cpu_count(),
    gpu_layers=1
)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(
    name="test",
    embedding_function=sentence_transformer_ef
)

query_string = st.text_input("Please enter your question here:", "Who is Gottlieb Daimler?")
question_tokens = preprocess_documents([query_string])

if not question_tokens:
    st.error("Question could not be preprocessed. Please try a different input.")
    st.stop()

str_question = " ".join(question_tokens[0]).strip()

results = collection.query(query_texts=[str_question], n_results=10)

doc = ""
for e in results["documents"]:
    for con in e:
        doc += con.strip() + "\n\n"

st.success('Retrieved context from ChromaDB')
st.write("### Retrieved Context")
st.write(doc)

final_prompt = PromptTemplate(
    template="""<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words just from the context.
Answer the question below from the context below in several sentences. You must remove the unrelated information also. If the provided information is not related, you must say that you can not answer based on the provided information:
{context}
{question} [/INST] </s>
""",
    input_variables=["question", "context"]
)

final_chain = final_prompt | llm
response = final_chain.invoke({"question": query_string, "context": doc})

gc.collect()
torch.cuda.empty_cache()

st.success('Answer generated successfully!')
st.write("### Answer")
st.write(response)