import os
import warnings
import gc
import json
import shutil
import csv
import re
import uuid
import base64
import time
import itertools
import numpy as np
import librosa
import tensorflow_hub as tfhub
import cv2
import torch
import subprocess
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from transformers import pipeline
from groq import Groq
from datetime import datetime
from typing import List, Optional, Literal, Any, Dict, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.getcwd(), "tfhub_cache")
os.makedirs(os.environ["TFHUB_CACHE_DIR"], exist_ok=True)

load_dotenv()

# --- 🔑 API KEY ROTATION SYSTEM ---
API_KEYS = []
if os.getenv("GROQ_API_KEY"): API_KEYS.append(os.getenv("GROQ_API_KEY"))
for i in range(1, 10):
    key = os.getenv(f"GROQ_API_KEY{i}")
    if key: API_KEYS.append(key)
API_KEYS = list(set([k for k in API_KEYS if k]))

if not API_KEYS:
    print("CRITICAL ERROR: No Groq API keys found! Check .env file.")
    exit(1)

print(f"Loaded {len(API_KEYS)} API Key(s) for rotation.")
KEY_CYCLE = itertools.cycle(API_KEYS)

def get_groq_client():
    return Groq(api_key=next(KEY_CYCLE))

# --- CONSTANTS ---
VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULT_DIR = os.path.join(DATA_DIR, "result")
JSON_SUMMARY_DIR = os.path.join(DATA_DIR, "jsonsummary")
TEMP_DIR = os.path.join(DATA_DIR, "temp_processing")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

for d in [INCOMING_DIR, PROCESSED_DIR, RESULT_DIR, JSON_SUMMARY_DIR, TEMP_DIR, CHROMA_DB_DIR]:
    os.makedirs(d, exist_ok=True)

# --- DATA MODELS ---

class FileMetadata(BaseModel):
    place: Optional[str] = "Unknown"
    date: Optional[str] = "Unknown"
    time: Optional[str] = "Unknown"

class AudioExtraction(BaseModel):
    gunshot_classification: Optional[str] = Field(None)
    times_detected: Optional[str] = Field(None)
    screams_panic: Optional[bool] = Field(None)
    background_noise: Optional[str] = Field(None)

class VisualExtraction(BaseModel):
    object_detection: List[str] = Field(default_factory=list)
    number_of_persons: Optional[int] = Field(None) # Renamed field
    threat_posture: Optional[str] = Field(None)
    force_identification: Optional[str] = Field(None) # New IFF field
    environment_clues: List[str] = Field(default_factory=list)
    detection_confidence: Optional[Any] = Field(None)

class TextExtraction(BaseModel):
    entities: Optional[Any] = Field(default_factory=list)
    locations: Optional[Any] = Field(default_factory=list)
    events: Optional[Any] = Field(default_factory=list)
    sentiment_urgency: Optional[Any] = Field(None)
    class Config: extra = 'allow'

class RakshakState(BaseModel):
    file_path: str
    file_name: str
    file_type: Literal["text", "audio", "video", "image", "unknown"] = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    file_metadata: Optional[FileMetadata] = Field(default_factory=FileMetadata)
    audio_data: Optional[AudioExtraction] = None
    visual_data: Optional[VisualExtraction] = None
    text_data: Optional[TextExtraction] = None
    error_message: Optional[str] = None
    summary: Optional[str] = None

# --- MEMORY ---

class RakshakMemory:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="rakshak_intel_logs")
    def store(self, text: str, metadata: dict):
        if not text: return
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        ids, embeddings, docs, metas = [], [], [], []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{metadata.get('source', 'unknown')}_{idx}_{uuid.uuid4().hex[:6]}"
            embeddings.append(self.embedder.encode(chunk).tolist())
            ids.append(chunk_id)
            docs.append(chunk)
            meta = metadata.copy()
            meta.update({'chunk_index': idx, 'timestamp': datetime.now().isoformat()})
            metas.append(meta)
        if ids: self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
memory = RakshakMemory()

# --- HELPER FUNCTIONS ---

def call_groq_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try: return func()
        except Exception as e:
            if attempt == max_retries - 1: raise e
            time.sleep(1)

def encode_image(image_path_or_array):
    if isinstance(image_path_or_array, str):
        with open(image_path_or_array, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        _, buffer = cv2.imencode('.jpg', image_path_or_array)
        return base64.b64encode(buffer).decode('utf-8')

def parse_filename_metadata(filename: str) -> FileMetadata:
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    meta = FileMetadata()
    if len(parts) >= 3:
        meta.time = parts[-1]
        meta.date = parts[-2]
        meta.place = "_".join(parts[:-2])
    elif len(parts) == 2:
        meta.place = parts[0]
        meta.date = parts[1]
    return meta

def extract_json_from_text(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group())
    except: pass
    return {}

def store_results_in_chromadb():
    print("\n>>> Storing Results in ChromaDB...")
    categories = ["video", "audio", "image", "text"]
    stored_count = 0
    for cat in categories:
        cat_result_dir = os.path.join(RESULT_DIR, cat)
        if not os.path.exists(cat_result_dir): continue
        for f in os.listdir(cat_result_dir):
            if f.endswith("_RESULT.json"):
                try:
                    with open(os.path.join(cat_result_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    text_parts = [f"File: {data.get('file_name')}"]
                    if data.get('summary'): text_parts.append(f"Summary: {data['summary']}")
                    full_text = "\n".join(text_parts)
                    meta = data.get('file_metadata', {})
                    metadata = {"source": data.get('file_name'), "type": cat, "location": meta.get('place', 'Unknown')}
                    memory.store(full_text, metadata)
                    stored_count += 1
                except Exception as e: print(f"   [Error storing {f}]: {e}")
    print(f"   ✓ Stored {stored_count} results in ChromaDB")

def generate_category_summaries():
    print("\n>>> Generating Consolidated JSON Summaries...")
    categories = ["video", "audio", "image", "text"]
    for cat in categories:
        cat_result_dir = os.path.join(RESULT_DIR, cat)
        if not os.path.exists(cat_result_dir): continue
        files_data = []
        for f in os.listdir(cat_result_dir):
            if f.endswith("_RESULT.json"):
                try:
                    with open(os.path.join(cat_result_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        files_data.append({
                            "file_name": data.get('file_name'),
                            "location": data.get('file_metadata', {}).get('place', 'Unknown'),
                            "summary": data.get('summary', 'No summary')[:500]
                        })
                except Exception as e: print(f"   [Error reading {f}]: {e}")
        
        if not files_data: continue
        
        files_info = "\n".join([f"- {item['file_name']} (Loc: {item['location']}): {item['summary']}" for item in files_data])
        prompt = f"""Analyze these {cat} files. Provide JSON: "overall_situation", "related_incidents", "individual_summaries", "threat_assessment". Files: {files_info}"""
        try:
            groq_client = get_groq_client()
            def summary_call(): return groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.3, max_tokens=1000)
            response = call_groq_with_retry(summary_call)
            analysis = json.loads(response.choices[0].message.content)
            output = {"report_type": f"consolidated_{cat}_analysis", "generated_at": datetime.now().isoformat(), "total_files": len(files_data), "analysis": analysis}
            out_path = os.path.join(JSON_SUMMARY_DIR, f"{cat}s_summary.json")
            with open(out_path, 'w', encoding='utf-8') as f: json.dump(output, f, indent=4)
            print(f"   [Report] Created {out_path}")
        except Exception as e: print(f"   [Error generating {cat} summary]: {e}")

# --- NODES ---

def classifier_node(state: RakshakState) -> RakshakState:
    ext = os.path.splitext(state.file_name)[1].lower()
    if ext in ['.txt', '.pdf', '.csv', '.md']: state.file_type = "text"
    elif ext in ['.mp3', '.wav', '.flac', '.m4a']: state.file_type = "audio"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']: state.file_type = "video"
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']: state.file_type = "image"
    else: state.file_type = "unknown"
    state.file_metadata = parse_filename_metadata(state.file_name)
    print(f"[{state.file_name}] Classified: {state.file_type} | Loc: {state.file_metadata.place}")
    return state

def text_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Text...")
    try:
        content = ""
        ext = os.path.splitext(state.file_path)[1].lower()
        if ext == ".pdf":
            with pymupdf.open(state.file_path) as doc:
                for page in doc: content += page.get_text() + "\n"
        elif ext in [".txt", ".md", ".log"]:
            with open(state.file_path, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
        content = content.strip()
        if not content: raise ValueError("No content")
        groq_client = get_groq_client()
        prompt = f"Analyze intel report. Content: '{content[:3000]}'. Extract JSON: summary, entities, locations, events, sentiment_urgency"
        def text_call():
            return groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        completion = call_groq_with_retry(text_call)
        analysis = json.loads(completion.choices[0].message.content)
        memory.store(content, {"source": state.file_name, "type": "text"})
        state.summary = analysis.get("summary", "No summary.")
        state.text_data = TextExtraction(**analysis)
    except Exception as e:
        print(f"   [Error] Text failed: {e}")
        state.error_message = str(e)
    return state

def audio_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Audio...")
    try:
        wav_data, sr = librosa.load(state.file_path, sr=16000, mono=True)
        rmse = librosa.feature.rms(y=wav_data)[0]
        volume_spike = np.max(rmse) > (np.mean(rmse) * 4)
        yamnet = tfhub.load('https://tfhub.dev/google/yamnet/1')
        class_names = [row[2] for row in csv.reader(open(yamnet.class_map_path().numpy()))][1:]
        scores, _, _ = yamnet(wav_data)
        detected = []
        has_speech = False
        gunshot = False
        scream = False
        for i in np.argsort(np.mean(scores, axis=0))[::-1][:5]:
            if np.mean(scores, axis=0)[i] > 0.10:
                sound = class_names[i]
                detected.append(sound)
                if "Speech" in sound: has_speech = True
                if any(x in sound for x in ["Gunshot", "Explosion", "Bang", "Blast", "Burst", "Fire"]): gunshot = True
                if "Scream" in sound: scream = True
        if volume_spike and not gunshot:
            detected.append("Impulse Event (Physics)")
            gunshot = True
        transcript = ""
        if has_speech:
            groq_client = get_groq_client()
            def whisper_call():
                with open(state.file_path, "rb") as f:
                    return groq_client.audio.transcriptions.create(file=f, model="whisper-large-v3-turbo", response_format="json")
            try: transcript = call_groq_with_retry(whisper_call).text
            except: pass
        state.audio_data = AudioExtraction(gunshot_classification="Detected" if gunshot else "None", screams_panic=scream, background_noise=", ".join(detected), times_detected=datetime.now().strftime("%H:%M:%S"))
        state.summary = f"Detected: {', '.join(detected)}. Transcript: {transcript[:500]}..."
        if transcript: memory.store(transcript, {"source": state.file_name, "type": "audio"})
    except Exception as e:
        print(f"   [Error] Audio failed: {e}")
        state.error_message = str(e)
    return state

# --- VIDEO NODE (VISION ONLY + IDENTITY) ---
def video_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Video (Llama Vision)...")
    try:
        cap = cv2.VideoCapture(state.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        max_people = 0
        all_objects = set()
        descriptions = []
        is_armed = False
        force_type_found = "Unknown"
        
        frame_count = 0
        groq_client = get_groq_client()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Analyze every 2 seconds
            if frame_count % 60 == 0:
                try:
                    base64_image = encode_image(frame)
                    current_client = get_groq_client()
                    
                    prompt = """
                    Analyze this surveillance frame.
                    
                    1. COUNT: People and Weapons.
                    2. IDENTIFY FORCE TYPE (Look at Uniforms):
                       - "Regular Military": Matching full camo, helmets, standard gear.
                       - "Irregular Hostile": Mixed civilian/camo, bandanas, non-standard gear.
                       - "Civilian": No military gear.
                    3. POSTURE: Passive vs Active.

                    Return JSON ONLY:
                    {
                        "person_count": int,
                        "weapons_detected": ["list"],
                        "force_type": "Regular Military" or "Irregular Hostile" or "Civilian",
                        "threat_level": "High/Med/Low",
                        "description": "brief summary"
                    }
                    """
                    
                    def vision_call():
                        return current_client.chat.completions.create(
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ],
                            model=VISION_MODEL_ID,
                            temperature=0.1, max_tokens=200
                        )

                    response = call_groq_with_retry(vision_call)
                    data = extract_json_from_text(response.choices[0].message.content)
                    
                    if data:
                        p_cnt = data.get("person_count", 0)
                        w_list = data.get("weapons_detected", [])
                        force_type = data.get("force_type", "Unknown")
                        
                        max_people = max(max_people, p_cnt)
                        all_objects.update(w_list)
                        if w_list: is_armed = True
                        if force_type != "Unknown": force_type_found = force_type
                        
                        desc = data.get("description", "")
                        descriptions.append(desc)
                        print(f"      [Vision] Pax: {p_cnt} | ID: {force_type} | Wpn: {w_list}")

                except Exception as e:
                    print(f"      [Vision Skipped] {e}")

            frame_count += 1
        cap.release()
        
        # Audio Extraction
        audio_path = os.path.join(TEMP_DIR, f"{state.file_name}.wav")
        subprocess.run(f'ffmpeg -y -i "{state.file_path}" -vn -ac 1 -ar 16000 "{audio_path}" -loglevel quiet', shell=True)
        detected_sounds, gunshot_detected, scream_detected = [], False, False
        if os.path.exists(audio_path):
            wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            if np.max(librosa.feature.rms(y=wav_data)[0]) > (np.mean(librosa.feature.rms(y=wav_data)[0]) * 4): gunshot_detected = True
            yamnet = tfhub.load('https://tfhub.dev/google/yamnet/1')
            class_names = [row[2] for row in csv.reader(open(yamnet.class_map_path().numpy()))][1:]
            for i in np.argsort(np.mean(yamnet(wav_data)[0], axis=0))[::-1][:5]:
                sound = class_names[i]
                detected_sounds.append(sound)
                if any(x in sound for x in ["Gunshot", "Explosion"]): gunshot_detected = True
            os.remove(audio_path)
            
        # Summary
        groq_client = get_groq_client()
        val_res = {"is_credible": True, "confidence": "Medium"}
        try:
            val_prompt = f"Validate Threat. Visuals: {list(set(descriptions))[:3]}. Weapons: {list(set(all_objects))}. Audio: {detected_sounds}."
            def val_call(): return groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": val_prompt}], response_format={"type": "json_object"})
            val_res = json.loads(call_groq_with_retry(val_call).choices[0].message.content)
        except: pass
        
        sys_prompt = "Military AI. Summarize the event."
        def summary_call(): return groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Vision: {list(set(descriptions))[:5]}."}], max_tokens=250)
        state.summary = call_groq_with_retry(summary_call).choices[0].message.content
        
        state.visual_data = VisualExtraction(
            object_detection=list(all_objects), 
            number_of_persons=max_people, 
            threat_posture="Armed" if is_armed else "Unarmed", 
            force_identification=force_type_found,
            environment_clues=list(set(descriptions))[:5], 
            detection_confidence=val_res.get('confidence', 'high')
        )
        state.audio_data = AudioExtraction(gunshot_classification="Detected" if gunshot_detected else "None", background_noise=", ".join(detected_sounds))
        
    except Exception as e:
        print(f"   [Error] Video failed: {e}")
        state.error_message = str(e)
    return state

# --- IMAGE NODE (VISION ONLY + IDENTITY) ---
def image_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Image (Llama Vision)...")
    try:
        base64_image = encode_image(state.file_path)
        groq_client = get_groq_client()
        
        prompt = """
        Analyze image. 
        1. COUNT People/Weapons.
        2. IDENTIFY FORCE TYPE (Uniforms): Regular Military / Irregular Hostile / Civilian.
        3. POSTURE: Passive vs Active.

        Return JSON ONLY:
        {
            "person_count": int,
            "weapons_detected": ["list"],
            "force_type": "string",
            "threat_level": "High/Med/Low",
            "description": "summary"
        }
        """
        
        def vision_call():
            return groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                model=VISION_MODEL_ID,
                temperature=0.1
            )
            
        response = call_groq_with_retry(vision_call)
        data = extract_json_from_text(response.choices[0].message.content)
        
        if not data: data = {"description": "Vision failed to parse"}

        state.summary = data.get("description")
        memory.store(state.summary, {"source": state.file_name, "type": "image"})
        
        state.visual_data = VisualExtraction(
            object_detection=data.get("weapons_detected", []),
            number_of_persons=data.get("person_count", 0),
            threat_posture="Armed" if data.get("weapons_detected") else "Unknown",
            force_identification=data.get("force_type", "Unknown"),
            environment_clues=[data.get("description")]
        )
    except Exception as e:
        print(f"   [Error] Image failed: {e}")
        state.error_message = str(e)
    return state

def save_node(state: RakshakState) -> RakshakState:
    output = state.model_dump(exclude={'file_path'})
    type_folder = os.path.join(RESULT_DIR, state.file_type)
    os.makedirs(type_folder, exist_ok=True)
    json_path = os.path.join(type_folder, f"{os.path.splitext(state.file_name)[0]}_RESULT.json")
    with open(json_path, 'w') as f: json.dump(output, f, indent=4)
    dest = os.path.join(PROCESSED_DIR, state.file_name)
    if os.path.exists(dest): os.remove(dest)
    shutil.move(state.file_path, dest)
    state.processing_status = "completed"
    print(f"[{state.file_name}] Saved & Archived.")
    return state

# --- WORKFLOW ---
workflow = StateGraph(RakshakState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("text", text_node)
workflow.add_node("audio", audio_node)
workflow.add_node("video", video_node)
workflow.add_node("image", image_node)
workflow.add_node("save", save_node)
workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", lambda x: x.file_type, {"text": "text", "audio": "audio", "video": "video", "image": "image", "unknown": "save"})
workflow.add_edge("text", "save")
workflow.add_edge("audio", "save")
workflow.add_edge("video", "save")
workflow.add_edge("image", "save")
workflow.add_edge("save", END)
app = workflow.compile()

# --- RUNNER ---

def process_single_file(filename: str):
    file_path = os.path.join(INCOMING_DIR, filename)
    try:
        result = app.invoke(RakshakState(file_path=file_path, file_name=filename))
        if isinstance(result, BaseModel): state_dict = result.model_dump()
        elif isinstance(result, dict): state_dict = result
        else: state_dict = result.__dict__
        error_msg = state_dict.get('error_message')
        return (filename, not error_msg, error_msg if error_msg else "Success")
    except Exception as e: return (filename, False, str(e))

def run_hybrid_system():
    print(">>> RAKSHAK SYSTEM STARTED (Vision Only + IFF Identity)")
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    if not files: 
        print(">>> No files in incoming directory.")
        generate_category_summaries()
        return
    videos = [f for f in files if os.path.splitext(f)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    others = [f for f in files if f not in videos]
    results = []
    if others:
        print(f"\n--- Processing {len(others)} Files (Parallel) ---")
        with ThreadPoolExecutor(max_workers=4) as ex:
            for fut in as_completed({ex.submit(process_single_file, f): f for f in others}):
                res = fut.result()
                results.append(res)
                print(f"   {'✓' if res[1] else '✗'} {res[0]}")
    if videos:
        print(f"\n--- Processing {len(videos)} Videos (Sequential) ---")
        for v in videos:
            res = process_single_file(v)
            results.append(res)
            print(f"   {'✓' if res[1] else '✗'} {res[0]}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    successful_count = sum(1 for _, s, _ in results if s)
    print(f"\n{'='*60}\nSUMMARY: {successful_count}/{len(files)} Successful\n{'='*60}")
    store_results_in_chromadb()
    generate_category_summaries()

if __name__ == "__main__":
    run_hybrid_system()