from asyncio import tasks
import asyncio
import os
import shutil
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.load import dumps, loads


# --- Docling Imports ---
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.chunking import HybridChunker

app = FastAPI(title="RAG DIA")

# --- 1. Initialization ---
print("Initializing FastAPI backend...")

# LLM Local (Llama 3.2 LM Studio)
llm = ChatOpenAI(
    model="llama-3.2-3b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key="not_required",
    temperature=0.1
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'}, # 'cuda' for GPU, 'cpu' for CPU
    encode_kwargs={'normalize_embeddings': True} # Normalize embeddings to unit length for better cosine similarity performance
)

embeddings._client.max_seq_length = 620 # Max tokens for BGE-M3 (it can handle up to 8192)

# ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="rag_collection",
    embedding_function=embeddings
)

# --- Variables ---
processed_files = set()
LAST_RETRIEVED_DOCS = []

# --- Pydantic model ---
class ChatRequest(BaseModel):
    message: str
    selected_files: List[str]
    chat_history: List[Dict[str, str]] = []


def reciprocal_rank_fusion(results: list[list], k=60):
    """RRF implementation to fuse results from multiple queries. Each result is a list of documents sorted by relevance."""
    fused_scores = {}
    doc_lookup = {}

    for docs in results:

        for rank, doc in enumerate(docs):

            source = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk_index", -1)

            # Unique ID
            doc_id = f"{source}_chunk_{chunk}"

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_lookup[doc_id] = doc

            fused_scores[doc_id] += 1.0 / (rank + k)

    # Sort by fused score
    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return unique docuements with their fused scores
    return [
        (doc_lookup[doc_id], score)
        for doc_id, score in reranked
    ]


# --- 2. Endpoints ---

@app.get("/files")
async def get_available_files():
    """Retrieve all unique source filenames from the ChromaDB collection to populate the file selection panel."""
    global processed_files
    try:
        # Get all metadata entries from the vectorstore
        db_data = vectorstore.get(include=["metadatas"])
        
        sources = set()
        for meta in db_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        
        processed_files = sources
        return {"files": list(sources)}
    except Exception as e:
        print(f"Error retrieving files from ChromaDB: {e}")
        return {"files": []}
    

@app.post("/upload")
async def process_files(files: List[UploadFile] = File(...)):
    """Process uploaded files with Docling and EasyOCR, split into chunks, and store in ChromaDB."""
    global processed_files
    
    new_docs = []
    new_filenames = []
    os.makedirs("temp_uploads", exist_ok=True)

    for file_obj in files:
        filename = file_obj.filename
        
        if filename in processed_files:
            continue 

        temp_path = os.path.join("temp_uploads", filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file_obj.file, buffer)

        try:
            print(f"Processing {filename} with Docling and EasyOCR...")
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True 
            pipeline_options.ocr_options = EasyOcrOptions() 
            
            doc_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            custom_chunker = HybridChunker(
                tokenizer="BAAI/bge-m3",
                max_tokens=600,
                overlap_tokens=200,
                merge_peers=True         # Merge chunks that are close together to preserve context
            )

            loader = DoclingLoader(
                file_path=temp_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=custom_chunker,
                converter=doc_converter
            )
            
            splits = loader.load()

            for i, doc in enumerate(splits):
                doc.metadata["source"] = filename
                doc.metadata["chunk_index"] = i
            
            new_docs.extend(splits)
            new_filenames.append(filename)
            processed_files.add(filename)
            print(f"Success: {filename} parsed into {len(splits)} chunks.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    status_message = ""
    if new_docs:
        # Filter complex metadata and assign deterministic IDs
        cleaned_docs = filter_complex_metadata(new_docs)
        doc_ids = [f"{doc.metadata['source']}_chunk_{doc.metadata['chunk_index']}" for doc in cleaned_docs]
        
        vectorstore.add_documents(documents=cleaned_docs, ids=doc_ids)
        status_message = f"Processed: {', '.join(new_filenames)}. Total chunks added: {len(cleaned_docs)}"
    else:
        status_message = "No new valid files were processed."

    return {"processed_files": list(processed_files), "status_message": status_message}


@app.post("/chat")
async def chat_response(request: ChatRequest):
    global LAST_RETRIEVED_DOCS
    
    if not request.message: 
        raise HTTPException(status_code=400, detail="Empty message")
    if not request.selected_files: 
        return {"response": "Please select at least one file from the panel."}
    
    history_str = ""
    for msg in request.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    
    if not history_str:
        history_str = "No previous history. This is the first question."

    print(f"\n--- Starting RAG-Fusion for: '{request.message}' ---")

    # --- 1. Multi-Query Generation ---
    mq_template = """
        You are an AI assistant. Generate 5 different versions of the given 
        user question to retrieve relevant documents from a vector database.

        By generating multiple perspectives, help overcome limitations of 
        distance-based similarity search.

        Chat history:
        {chat_history}

        User question:
        {question}

        Output only the alternative questions, one per line.

    """
    
    prompt_mq = PromptTemplate.from_template(mq_template)
    mq_chain = prompt_mq | llm | StrOutputParser()
    
    generated_queries_str = mq_chain.invoke({
        "question": request.message, 
        "chat_history": ""
    })

    queries = [request.message] + [q.strip() for q in generated_queries_str.split('\n') if q.strip()]
    print(f"Generated queries:\n{queries}")

    # --- 2. Parallel recovery ---
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"source": {"$in": request.selected_files}}}
    )
    
    all_retrieved_results = []
    tasks = [retriever.ainvoke(q) for q in queries]
    all_retrieved_results = await asyncio.gather(*tasks)

    # --- 3. Reciprocal Rank Fusion (RRF) ---
    fused_docs = reciprocal_rank_fusion(all_retrieved_results)
    final_top_docs = [doc for doc, score in fused_docs[:6]]
    LAST_RETRIEVED_DOCS = final_top_docs

    print(f"Fusion completed. {len(final_top_docs)} unique chunks selected for the response.\n")

    print(f"Final retrived chunks (after RRF):")
    for i, doc in enumerate(final_top_docs):
        print(f"  {i+1}. {doc.metadata.get('source', 'Unknown')} - Chunk {doc.metadata.get('chunk_index', 0)}")

    # --- 4. Generation of response ---
    template_qa = (
        "You are a question-answering assistant.\n"
        "Answer the user's question using ONLY the information provided in the context.\n"
        "Rules:\n"
        "- Do NOT use any external knowledge.\n"
        "- If the answer is not explicitly stated in the context, say that you don't know.\n"
        "- Do NOT guess or infer missing information.\n"
        "- Be concise and precise.\n\n"
        
        "Chat History:\n"
        "{chat_history}\n\n"
        
        "Context:\n"
        "{context}\n\n"
        
        "Question:\n"
        "{question}\n\n"
        
        "Answer:\n"
    )
    
    prompt_qa = ChatPromptTemplate.from_template(template_qa)
    qa_chain = prompt_qa | llm | StrOutputParser()
    
    context_text = "\n\n".join(doc.page_content for doc in final_top_docs)

    response = qa_chain.invoke({
        "context": context_text,
        "question": request.message,
        "chat_history": history_str
    })
    
    return {"response": response}


@app.get("/inspector")
async def visualize_extended_context():
    global LAST_RETRIEVED_DOCS
    
    if not LAST_RETRIEVED_DOCS:
        return {"html": "<div style='padding:20px; text-align:center;'>No context available. Please ask a question in the chat first.</div>"}

    html_output = f"<h3 style='margin-bottom:20px; color: #333;'>Context used for the last response</h3>"

    for i, doc in enumerate(LAST_RETRIEVED_DOCS):
        source = doc.metadata.get("source", "Unknown")
        idx = doc.metadata.get("chunk_index", 0)
        
        # Search for previous and next chunks in the database
        prev_id = f"{source}_chunk_{idx - 1}"
        next_id = f"{source}_chunk_{idx + 1}"
        
        try:
            vecinos_data = vectorstore._collection.get(ids=[prev_id, next_id])
            
            text_prev = "<i>(Start of document / No context)</i>"
            text_next = "<i>(End of document / No context)</i>"
            
            if vecinos_data and "ids" in vecinos_data:
                for j, doc_id in enumerate(vecinos_data["ids"]):
                    if doc_id == prev_id:
                        text_prev = vecinos_data["documents"][j]
                    elif doc_id == next_id:
                        text_next = vecinos_data["documents"][j]
        except Exception as e:
            text_prev = f"<i>(Error loading previous context: {e})</i>"
            text_next = f"<i>(Error loading next context: {e})</i>"

        text_curr = doc.page_content

        html_output += f"""
        <div style="border: 1px solid #ccc; border-radius: 8px; margin-bottom: 30px; overflow: hidden; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background-color: white;">
            <div style="background-color: #e0e0e0; padding: 8px 15px; border-bottom: 1px solid #999; font-weight: bold; font-size: 0.9em; color: #000000;">
                Rank #{i+1} | File: {source} | Chunk ID: {idx}
            </div>
            <div style="background-color: #fff3e0; padding: 10px; font-size: 0.85em; color: #444444; border-bottom: 1px dotted #ccc;">
                <strong style="color: #d84315;">Previous context:</strong><br>{text_prev}
            </div>
            <div style="background-color: #f1f8e9; padding: 15px; font-size: 1em; border-left: 5px solid #4caf50; color: #000000;">
                <strong style="color: #2e7d32;">Retrieved chunk:</strong><br>{text_curr}
            </div>
            <div style="background-color: #e3f2fd; padding: 10px; font-size: 0.85em; color: #444444; border-top: 1px dotted #ccc;">
                <strong style="color: #1565c0;">Next context:</strong><br>{text_next}
            </div>
        </div>
        """
    return {"html": html_output}