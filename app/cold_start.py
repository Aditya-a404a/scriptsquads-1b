import json
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

RETRIEVER_MODEL = './models/bge-small-en-v1.5'
RERANKER_MODEL = './models/ms-marco-MiniLM-L-6-v2'

SUMMARIZATION_MODEL = './models/text_summarization'
DEVICE = "cpu"
summarizer = pipeline("summarization", model="./models/text_summarization")



def load_models():
    """Loads all necessary models with detailed progress."""
    print(" -> Loading retriever model (bge-small)...")
    retriever_model = SentenceTransformer(RETRIEVER_MODEL, device=DEVICE)
    print("    Retriever loaded.")
    print(" -> Loading re-ranker model (cross-encoder)...")
    reranker_model = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    print("    Re-ranker loaded.")
    print(" -> Loading summarizer model (Falconsai T5)...")
    summarizer_model = T5ForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL).to(DEVICE)
    summarizer_tokenizer = T5Tokenizer.from_pretrained(SUMMARIZATION_MODEL)
    print("    Summarizer loaded.")
    return retriever_model, reranker_model, summarizer_model, summarizer_tokenizer

def chunk_data(raw_data):
    """
    Groups the flat list of text elements into logical sections,
    ensuring that 'List-item' and 'Text' are appended correctly
    and that no chunks with empty content are created.
    """
    chunks = []
    current_chunk = None
    # Filter out any items that don't have text content to begin with
    filtered_data = [item for item in raw_data if item.get("text", "").strip()]

    for item in filtered_data:
        if item.get("class") == "Section-header":
            # Before starting a new chunk, check if the previous one is valid
            # A valid chunk must exist and have non-empty content.
            if current_chunk and current_chunk["content"].strip():
                chunks.append(current_chunk)
            
           
            current_chunk = {
                "file_name": item["file_name"],
                "page_number": item["page"],
                "section_title": item["text"].strip(),
                "content": ""  # Initialize content as empty
            }
        # If it's not a header and we have an active chunk, it's content
        elif current_chunk and item.get("text"):
            # This will append both 'Text' and 'List-item' content
            current_chunk["content"] += item["text"].strip() + " "

    
    if current_chunk and current_chunk["content"].strip():
        chunks.append(current_chunk)

    # Final cleanup of trailing spaces for all chunks
    for chunk in chunks:
        chunk["content"] = chunk["content"].strip()

    return chunks

def find_top_candidates(chunks, persona, jtbd, retriever_model, top_k=50):
    """Finds the most likely candidates using a fast bi-encoder."""
    enhanced_query = f"{persona} {jtbd}"
    query_embedding = retriever_model.encode(enhanced_query, convert_to_tensor=True)
    chunk_contents = [chunk["content"] for chunk in chunks]
    chunk_embeddings = retriever_model.encode(chunk_contents, convert_to_tensor=True, normalize_embeddings=True)
    similarities = cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = torch.topk(similarities, k=min(top_k, len(chunks))).indices
    top_candidates = [chunks[i] for i in top_indices]
    return top_candidates, enhanced_query

def rerank_with_cross_encoder(query, candidates, reranker_model):
    """Re-ranks the top candidates using a more accurate cross-encoder."""
    pairs = [[query, candidate['content']] for candidate in candidates]
    scores = reranker_model.predict(pairs)
    for i in range(len(candidates)):
        candidates[i]['rerank_score'] = scores[i]
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

def generate_refined_text(chunks_to_summarize, model, tokenizer):
    """Generates a summary using a specialized T5 summarizer."""
    summarized_chunks = []
    for chunk in chunks_to_summarize:
        # --- MODIFIED: Simple prompt for a specialized summarizer ---
        prompt = chunk['content']
        # summarizer = pipeline("summarization", model="./models/text_summarization")
        # input_ids = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
        # summary_ids = model.generate(
        #     **input_ids,
        #     max_length=150,
        #     min_length=40,
        #     num_beams=4,
        #     early_stopping=True
        # )
        # refined_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        refined_text = summarizer(prompt, max_length=1000, min_length=30, do_sample=False)[0]["summary_text"]
        chunk_copy = chunk.copy()
        chunk_copy['refined_text'] = refined_text
        summarized_chunks.append(chunk_copy)
    return summarized_chunks


app = FastAPI(title="Persona-Driven Document Intelligence API")

class AnalysisRequest(BaseModel):
    persona: str
    job_to_be_done: str
    input_path: str
    output_directory: str

print("Starting server and loading models (this will take a moment)...")
RETRIEVER, RERANKER, SUMMARIZER, TOKENIZER = load_models()
print("\n✅ Server is ready to accept requests at http://127.0.0.1:8000")

@app.post("/analyze")
def analyze_documents(request: AnalysisRequest):
    """Receives a request, performs analysis, and returns the results."""
    print(f"Received request for persona: {request.persona}")
    try:
        with open(request.input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        chunks = chunk_data(raw_data)
    except FileNotFoundError: return {"error": "Input file not found", "path": request.input_path}
    except Exception as e: return {"error": "Failed to read or process input file", "details": str(e)}

    top_candidates, query = find_top_candidates(chunks, request.persona, request.job_to_be_done, RETRIEVER, top_k=50)
    filtered_candidates = top_candidates
    final_ranked_chunks = rerank_with_cross_encoder(query, filtered_candidates, RERANKER)
    chunks_for_summary = final_ranked_chunks[:5]
    
    
    summarized_sections = generate_refined_text(chunks_for_summary, SUMMARIZER, TOKENIZER)
    
    output_data = {
        "metadata": {"input_documents": list(set([item['file_name'] for item in raw_data])), "persona": request.persona, "job_to_be_done": request.job_to_be_done, "processing_timestamp": datetime.now().isoformat()},
        "extracted_section": [], "sub-section_analysis": []
    }

    for i, chunk in enumerate(final_ranked_chunks[:5]):
        output_data["extracted_section"].append({"document": chunk["file_name"],"section_title": chunk["section_title"], "importance_rank": i + 1,"page_number": chunk["page_number"]})
        
    for chunk in summarized_sections:
        output_data["sub-section_analysis"].append({"document": chunk["file_name"], "refined_text": chunk["refined_text"],"page_number": chunk["page_number"]})
    
    os.makedirs(request.output_directory, exist_ok=True)
    output_filename = "challenge1b_output.json"
    output_filepath = os.path.join(request.output_directory, output_filename)
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete. Output saved to {output_filepath}")
    return {"status": "success", "output_file": output_filepath}