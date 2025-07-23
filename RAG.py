import os
import gc
import torch
import chromadb
import streamlit as st
from gensim.parsing.preprocessing import preprocess_documents
from chromadb.utils import embedding_functions
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# === Load LLM
llm = CTransformers(
    model='TheBloke/Mistral-7B-Instruct-v0.1-GGUF',
    model_file='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    config={'max_new_tokens': 4096, 'temperature': 0.2, 'context_length': 8000},
    threads=os.cpu_count(),
    gpu_layers=1
)

# === Setup ChromaDB and embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(name="test", embedding_function=sentence_transformer_ef)

# === UI
st.title("🧠 Retrieval-Augmented Question Answering")
query_string = st.text_input("🔎 Please enter your question here:")

if st.button("Ask"):
    if query_string.strip():
        # === Preprocess query and retrieve docs
        question_tokens = preprocess_documents([query_string])[0]
        str_question = " ".join(question_tokens).strip()

        results = collection.query(query_texts=[str_question], n_results=10)

        # === Format retrieved documents
        doc = ""
        for e in results["documents"]:
            for con in e:
                doc += con.strip() + "\n\n"

        st.success('✅ Retrieved context from ChromaDB')
        st.subheader("📚 Retrieved Context")
        for idx, e in enumerate(results["documents"]):
            for item in e:
                st.markdown(f"{idx+1}. {item}")

        # === Prompt + LLM
        final_prompt = PromptTemplate(
            template="""
<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words just from the context.
Answer the question below from the context below in several sentences. You must remove the unrelated information also. 
If the provided information is not related, you must say that you can not answer based on the provided information:
{context}
{question} [/INST] </s>
""",
            input_variables=["question", "context"]
        )

        final_chain = final_prompt | llm
        response = final_chain.invoke({"question": query_string, "context": doc})

        st.success("✅ Answer generated successfully!")
        st.subheader("🤖 Answer")
        st.write(response)
    else:
        st.warning("Please enter a valid question.")

# === Cleanup
gc.collect()
torch.cuda.empty_cache()
