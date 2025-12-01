import os
import warnings
import gc

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

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
from ultralytics import YOLO
from transformers import pipeline
from groq import Groq
from datetime import datetime
from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- 🔑 API KEY ROTATION SYSTEM ---
API_KEYS = []

if os.getenv("GROQ_API_KEY"):
    API_KEYS.append(os.getenv("GROQ_API_KEY"))

for i in range(1, 10):
    key = os.getenv(f"GROQ_API_KEY{i}")
    if key:
        API_KEYS.append(key)

API_KEYS = list(set([k for k in API_KEYS if k]))

if not API_KEYS:
    print("❌ CRITICAL ERROR: No Groq API keys found! Check .env file.")
    exit(1)

print(f"✅ Loaded {len(API_KEYS)} API Key(s) for rotation.")
KEY_CYCLE = itertools.cycle(API_KEYS)

def get_groq_client():
    next_key = next(KEY_CYCLE)
    return Groq(api_key=next_key)

# --- CONSTANTS ---
VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct"

# ----------------------------------

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    if weights_only is None: weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULT_DIR = os.path.join(DATA_DIR, "result")
JSON_SUMMARY_DIR = os.path.join(DATA_DIR, "jsonsummary")
TEMP_DIR = os.path.join(DATA_DIR, "temp_processing")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

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
    number_of_attackers: Optional[int] = Field(None)
    threat_posture: Optional[str] = Field(None)
    environment_clues: List[str] = Field(default_factory=list)
    detection_confidence: Optional[Any] = Field(None)

class TextExtraction(BaseModel):
    entities: List[Any] = Field(default_factory=list)
    locations: List[Any] = Field(default_factory=list)
    events: List[Any] = Field(default_factory=list)
    sentiment_urgency: Optional[Any] = Field(None)
    
    class Config:
        extra = 'allow'

class RakshakState(BaseModel):
    file_path: str
    file_name: str
    file_type: Literal["text", "audio", "video", "image", "unknown"] = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    file_metadata: Optional[FileMetadata] = Field(default_factory=FileMetadata)
    
    audio_data: Optional[AudioExtraction] = None
    visual_data: Optional[VisualExtraction] = None
    text_data: Optional[TextExtraction] = None
    
    processing_status: str = "pending"
    error_message: Optional[str] = None
    summary: Optional[str] = None

# --- MEMORY ---

class RakshakMemory:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="rakshak_intel_logs")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def store(self, text: str, metadata: dict):
        if not text: return
        chunks = self.splitter.split_text(text)
        ids, embeddings, docs, metas = [], [], [], []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{metadata.get('source', 'unknown')}_{idx}_{uuid.uuid4().hex[:6]}"
            embeddings.append(self.embedder.encode(chunk).tolist())
            ids.append(chunk_id)
            docs.append(chunk)
            meta = metadata.copy()
            meta.update({'chunk_index': idx, 'timestamp': datetime.now().isoformat()})
            metas.append(meta)
            
        if ids:
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

memory = RakshakMemory()

# --- YOLO SETUP ---
print("🔄 Loading YOLO models...")
YOLO_PERSON = YOLO(PERSON_MODEL_PATH)
YOLO_WEAPON = YOLO(WEAPON_MODEL_PATH) if os.path.exists(WEAPON_MODEL_PATH) else None

if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    YOLO_PERSON.to('cuda')
    if YOLO_WEAPON: YOLO_WEAPON.to('cuda')
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️  No GPU - using CPU")
print("✅ YOLO models loaded\n")

# --- HELPER FUNCTIONS ---

def call_groq_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1: raise e
            time.sleep(1)

def encode_image(image_path_or_array):
    if isinstance(image_path_or_array, str):
        with open(image_path_or_array, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
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

def generate_category_summaries():
    """
    Generates consolidated JSON summary files for each category.
    This creates structured output suitable for dashboards/analysis.
    """
    print("\n>>> Generating Consolidated JSON Summaries...")
    
    categories = {
        "video": {"report_type": "consolidated_video_report", "incidents": []},
        "image": {"report_type": "consolidated_image_report", "incidents": []},
        "audio": {"report_type": "consolidated_audio_report", "incidents": []},
        "text":  {"report_type": "consolidated_text_report", "incidents": []}
    }

    # Iterate through all result JSONs
    for f in os.listdir(RESULT_DIR):
        if f.endswith("_RESULT.json"):
            try:
                with open(os.path.join(RESULT_DIR, f), 'r') as file:
                    data = json.load(file)
                    ftype = data.get('file_type')
                    
                    if ftype in categories:
                        # Build the structured incident object
                        incident = {
                            "file_name": data.get('file_name'),
                            "timestamp": data.get('timestamp'),
                            "location_metadata": data.get('file_metadata', {}),
                            "detailed_summary": data.get('summary', "No summary available."),
                            "threat_level": "Unknown",
                            "key_findings": []
                        }

                        # Extract type-specific details for better "Key Findings"
                        if ftype in ["video", "image"] and data.get("visual_data"):
                            vis = data["visual_data"]
                            incident["threat_level"] = vis.get("threat_posture", "Unknown")
                            incident["key_findings"] = vis.get("object_detection", [])
                        
                        elif ftype == "audio" and data.get("audio_data"):
                            aud = data["audio_data"]
                            incident["threat_level"] = "High" if aud.get("gunshot_classification") == "Detected" else "Low"
                            incident["key_findings"] = [aud.get("background_noise")]
                        
                        elif ftype == "text" and data.get("text_data"):
                            txt = data["text_data"]
                            incident["threat_level"] = str(txt.get("sentiment_urgency", "Unknown"))
                            # Flatten entities for key findings
                            if isinstance(txt.get("entities"), list):
                                incident["key_findings"] = txt.get("entities")[:5] # Top 5 entities

                        categories[ftype]["incidents"].append(incident)
                        
            except Exception as e:
                print(f"   [Error reading {f}]: {e}")

    # Write the consolidated JSON files
    for cat, content in categories.items():
        if content["incidents"]:
            content["generated_at"] = datetime.now().isoformat()
            content["total_files"] = len(content["incidents"])
            
            out_path = os.path.join(JSON_SUMMARY_DIR, f"{cat}s_summary.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4)
            print(f"   [Report] Created dajson/{cat}s_summary.json")

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
            return groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
        
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
        onset_env = librosa.onset.onset_strength(y=wav_data, sr=sr)
        volume_spike = np.max(rmse) > (np.mean(rmse) * 4)
        sudden_onset = np.max(onset_env) > 5.0
        high_energy_event = volume_spike and sudden_onset
        if high_energy_event: print(f"      [Audio Physics] High Energy Impulse detected")

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

        if high_energy_event and not gunshot:
            detected.append("Impulse Event (Physics)")
            gunshot = True

        transcript = ""
        if has_speech:
            groq_client = get_groq_client()
            def whisper_call():
                with open(state.file_path, "rb") as f:
                    return groq_client.audio.transcriptions.create(
                        file=f, model="whisper-large-v3-turbo", response_format="json"
                    )
            try:
                transcript = call_groq_with_retry(whisper_call).text
            except: pass

        state.audio_data = AudioExtraction(
            gunshot_classification="Detected" if gunshot else "None",
            screams_panic=scream,
            background_noise=", ".join(detected),
            times_detected=datetime.now().strftime("%H:%M:%S")
        )
        state.summary = f"Detected: {', '.join(detected)}. Transcript: {transcript[:50]}..."
        if transcript: memory.store(transcript, {"source": state.file_name, "type": "audio"})
    except Exception as e:
        print(f"   [Error] Audio failed: {e}")
        state.error_message = str(e)
    return state

def video_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Video...")
    try:
        cap = cv2.VideoCapture(state.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        detected_objs = []
        max_people = 0
        weapon_detected = False
        descriptions = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Analyze every ~2 seconds
            if frame_count % 60 == 0:
                p_count = len(YOLO_PERSON.predict(frame, classes=[0], verbose=False, conf=0.25)[0].boxes)
                max_people = max(max_people, p_count)
                
                w_names = []
                if YOLO_WEAPON:
                    w_res = YOLO_WEAPON.predict(frame, verbose=False, conf=0.60)
                    for box in w_res[0].boxes:
                        name = w_res[0].names[int(box.cls[0])]
                        w_names.append(name)
                        detected_objs.append(name)
                        weapon_detected = True
                
                if p_count > 0 or w_names or (frame_count % 120 == 0):
                    try:
                        base64_image = encode_image(frame)
                        groq_client = get_groq_client()
                        def vision_call():
                            return groq_client.chat.completions.create(
                                messages=[
                                    {"role": "user", "content": [
                                        {"type": "text", "text": "Analyze this surveillance frame for security threats. Detect: Armed personnel, Weapons, Bunkers, Smoke."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                    ]}
                                ],
                                model=VISION_MODEL_ID,
                                temperature=0.1, max_tokens=100
                            )
                        desc = call_groq_with_retry(vision_call).choices[0].message.content
                        descriptions.append(desc)
                        print(f"      [Vision] Time {frame_count/fps:.1f}s | {desc[:60]}...")
                    except Exception as e:
                        print(f"      [Vision Skipped] Network/Rate Limit")

            frame_count += 1
        cap.release()
        
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
                if any(x in sound for x in ["Gunshot", "Explosion", "Bang", "Blast", "Burst", "Fire"]): gunshot_detected = True
                if "Scream" in sound: scream_detected = True
            os.remove(audio_path)

        groq_client = get_groq_client()
        val_res = {"is_credible": True, "confidence": "Medium"}
        try:
            val_prompt = f"Validate Threat. Visuals: {list(set(descriptions))[:3]}. Weapons: {list(set(detected_objs))}. Audio: {detected_sounds}. Explosion: {gunshot_detected}. Output JSON: is_credible, confidence, reasoning"
            def val_call():
                return groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": val_prompt}],
                    response_format={"type": "json_object"}
                )
            val_res = json.loads(call_groq_with_retry(val_call).choices[0].message.content)
        except: pass

        groq_client = get_groq_client()
        def summary_call():
            return groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Military AI. Provide a detailed, elaborative tactical summary of the event."},
                    {"role": "user", "content": f"Vision: {list(set(descriptions))[:5]}. Audio: {detected_sounds}."}
                ], max_tokens=250
            )
        state.summary = call_groq_with_retry(summary_call).choices[0].message.content

        threat_level = "Armed" if weapon_detected or any("firing" in d.lower() for d in descriptions) else "Unarmed"
        state.visual_data = VisualExtraction(
            object_detection=list(set(detected_objs)),
            number_of_attackers=max_people,
            threat_posture=f"{threat_level} ({val_res.get('confidence', 'med')})",
            environment_clues=list(set(descriptions))[:5],
            detection_confidence=val_res.get('confidence', 'high')
        )
        state.audio_data = AudioExtraction(
            gunshot_classification="Detected" if gunshot_detected else "None",
            screams_panic=scream_detected,
            background_noise=", ".join(detected_sounds)
        )
    except Exception as e:
        print(f"   [Error] Video failed: {e}")
        state.error_message = str(e)
    return state

def image_node(state: RakshakState) -> RakshakState:
    print(f"[{state.file_name}] Processing Image...")
    try:
        img = cv2.imread(state.file_path)
        p_count = len(YOLO_PERSON.predict(img, classes=[0], verbose=False, conf=0.35)[0].boxes)
        objs = [f"{p_count} Person(s)"] if p_count else []
        weapon_count = 0
        if YOLO_WEAPON:
            res_w = YOLO_WEAPON.predict(img, verbose=False, conf=0.65)
            weapon_count = len(res_w[0].boxes)
            objs.extend([res_w[0].names[int(c)] for c in res_w[0].boxes.cls])
            
        try:
            base64_image = encode_image(state.file_path)
            groq_client = get_groq_client()
            def vision_call():
                return groq_client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "Analyze this surveillance frame for security threats. Detect: Armed personnel, Weapons, Bunkers, Smoke."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ]}
                    ],
                    model=VISION_MODEL_ID, temperature=0.1, max_tokens=100
                )
            desc = call_groq_with_retry(vision_call).choices[0].message.content
        except Exception as e:
            desc = "Vision analysis unavailable"

        state.summary = desc
        memory.store(desc, {"source": state.file_name, "type": "image"})
        state.visual_data = VisualExtraction(
            object_detection=objs,
            number_of_attackers=p_count,
            threat_posture="Armed" if weapon_count else "Unknown",
            environment_clues=[desc]
        )
    except Exception as e:
        print(f"   [Error] Image failed: {e}")
        state.error_message = str(e)
    return state

def save_node(state: RakshakState) -> RakshakState:
    output = state.model_dump(exclude={'file_path'})
    json_path = os.path.join(RESULT_DIR, f"{os.path.splitext(state.file_name)[0]}_RESULT.json")
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
workflow.add_conditional_edges("classifier", lambda x: x.file_type, 
    {"text": "text", "audio": "audio", "video": "video", "image": "image", "unknown": "save"})
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
        app.invoke(RakshakState(file_path=file_path, file_name=filename))
        return (filename, True, "Success")
    except Exception as e:
        return (filename, False, str(e))

def run_hybrid_system():
    print(">>> RAKSHAK SYSTEM STARTED (Llama 4 Vision + Multi-Key Rotation)")
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    if not files: print(">>> No files."); return
    
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
    
    if successful_count > 0:
        generate_category_summaries()

if __name__ == "__main__":
    run_hybrid_system()