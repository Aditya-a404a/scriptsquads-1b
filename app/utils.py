import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import json
import requests
# Removed multiprocessing import

# --- Configuration ---
MODEL_PATH = "./models/secretRecipe.pt"
API_ENDPOINT = "http://127.0.0.1:8000/analyze"
ROOT_DIR = "./" # The directory containing "Collection 1", "Collection 2", etc.
DPI = 120

# --- MODIFIED: process_pdf now accepts the loaded model as an argument ---
def process_pdf(pdf_path, torch_model):
    """
    This worker function now processes a single PDF using a pre-loaded model.
    """
    fname = os.path.basename(pdf_path)
    print(f"  -> Processing: {fname}")

    doc = fitz.open(pdf_path)
    ans = {"title": "", "outline": []}

    page_images = []
    for page in doc:
        pix = page.get_pixmap(dpi=DPI)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            image = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            image = arr[:, :, :3]
        page_images.append(image)

    if not page_images:
        doc.close()
        return None

    all_results = torch_model(page_images, conf=0.35, iou=0.8)

    all_headers = []
    for page_index, results in enumerate(all_results):
        page = doc[page_index]
        mat = fitz.Matrix(72/DPI, 72/DPI)
        page_words = page.get_text("words")

        yolo_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        yolo_classes = results.boxes.cls.cpu().numpy().astype(int)

        for i, yolo_box in enumerate(yolo_boxes):
            class_name = torch_model.names[yolo_classes[i]]
            text = ""
            yolo_pixel_rect = fitz.Rect(*yolo_box)
            yolo_pdf_rect = yolo_pixel_rect * mat
            words_in_box = [w[4] for w in page_words if fitz.Rect(w[:4]).intersects(yolo_pdf_rect)]
            if words_in_box:
                text = " ".join(words_in_box)

            if text:
                all_headers.append({
                    "class":class_name,
                    "file_name": fname,
                    "page": page_index + 1,
                    "text": text,
                    "box": yolo_box.tolist(),
                    "area": abs(yolo_box[3] - yolo_box[1])
                })

    if all_headers:
        final_outline = []
        for h in sorted(all_headers, key=lambda x: (x["page"], x["box"][1])):
            final_outline.append({"file_name":h["file_name"],"class":h["class"], "text": h["text"], "page": h["page"]})
        ans["outline"] = final_outline

    doc.close()
    return ans["outline"]

def process_collection(collection_path):
    """
    Processes a single collection sequentially.
    """
    print(f"\n--- Starting processing for collection: {os.path.basename(collection_path)} ---")

    pdf_dir = os.path.join(collection_path, "PDFs")
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    processed_data_path = os.path.join(collection_path, "processed_pdf_data.json")

    if not os.path.isdir(pdf_dir):
        print(f"Warning: 'PDFs' directory not found in {collection_path}. Skipping.")
        return

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Skipping.")
        return

    # --- MODIFIED: Load the YOLO model once for the entire collection ---
    print("Loading YOLO model for this collection...")
    yolo_model = YOLO(MODEL_PATH)
    print("Model loaded.")

    # --- MODIFIED: Process PDFs sequentially using a loop ---
    all_pdfs_data = []
    for pdf_file in pdf_files:
        result = process_pdf(pdf_file, yolo_model)
        if result:
            all_pdfs_data.append(result)

    # Flatten the list of lists into a single list
    processed_data = [item for sublist in all_pdfs_data for item in sublist]

    with open(processed_data_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"PDF data processed and saved to: {processed_data_path}")

    # Call the API server (no changes to this part)
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        api_payload = {
            "persona": input_data.get("persona").get("role"),
            "job_to_be_done": input_data.get("job_to_be_done").get("task"),
            "input_path": processed_data_path,
            "output_directory": collection_path
        }
        print(f"Calling analysis API for {os.path.basename(collection_path)}...")
        response = requests.post(API_ENDPOINT, json=api_payload)
        response.raise_for_status()
        print(f"✅ API call successful. Server response: {response.json()}")
    except FileNotFoundError:
        print(f"Error: {input_json_path} not found. Cannot call API.")
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}. Is the api_server.py running?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Finds all collection directories and processes each one sequentially.
    """
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Root directory '{ROOT_DIR}' not found.")
        return

    collections = [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

    if not collections:
        print(f"No collection folders found in '{ROOT_DIR}'.")
        return

    for collection_path in collections:
        process_collection(collection_path)

if __name__ == "__main__":
    main()