import os
import gc
import torch
from gensim.parsing.preprocessing import preprocess_documents
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from cleansing import cleansing

# Setup paths
DB_PATH = os.path.join(".", "db")
TEXT_PATH = os.path.join(".", "try", "test.txt")

if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"❌ File not found at: {TEXT_PATH}")

# Initialize ChromaDB
client = PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="test")

# Embedding model
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded")

# Insert logic
count = 0
cleaned_lines = cleansing(TEXT_PATH)
if not cleaned_lines:
    raise ValueError("❌ Cleansing returned no data. Check delimiter format in test.txt")

for data in cleaned_lines:
    doc = data.split("\n")
    documents = ""
    for line in doc:
        documents += line + ("\n\n" if "?" not in line else "")

    splits = [s.replace("\n", " ").replace("?", "?\n").strip() for s in documents.strip().split("\n\n") if s.strip()]

    for v in splits:
        newv = v.split(') \n')[0] if ') \n' in v else v
        newv = newv.replace("(", "").replace(")", "")
        for sep in [" Answer", "Answer ", ".  A", "A: "]:
            if sep in newv:
                newv = newv.split(sep)[0]
        newv = newv.strip()

        question = preprocess_documents([newv])[0]
        str_question = " ".join(question).strip()

        embedding = sentence_transformer_ef([str_question])
        if len(v) > 10 and "?" in v:
            collection.upsert(
                embeddings=embedding,
                documents=[v],
                metadatas=[{"source": ""}],
                ids=[str(count)]
            )
            print(f"[INFO] Inserted QA ID: {count}")
            count += 1

    if 'embedding' in locals():
        del embedding
    gc.collect()
    torch.cuda.empty_cache()

print("✅ All data inserted successfully.")
