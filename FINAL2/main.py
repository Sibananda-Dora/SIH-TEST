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
import numpy as np
import librosa
import tensorflow_hub as tfhub
import cv2
import torch
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from groq import Groq
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from pypdf import PdfReader
from dateutil import parser
import chromadb
from sentence_transformers import SentenceTransformer

# Try importing RecursiveCharacterTextSplitter, fallback if not available
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Simple fallback implementation
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text):
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

# --- SECURITY PATCH FOR PYTORCH 2.6+ ---
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                 weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INCOMING_DIR = os.path.join(DATA_DIR, "incoming")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULT_DIR = os.path.join(DATA_DIR, "result")
TEMP_DIR = os.path.join(DATA_DIR, "temp_processing")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
GROQ_API_KEY = "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"

# Model Paths (Relative to FINAL2)
WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Ensure directories exist
os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# --- 1. DETAILED STATE DEFINITION (Based on your Images) ---

class AudioExtraction(BaseModel):
    gunshot_classification: Optional[str] = Field(None, description="AK-series, pistol, 7.62mm or any other guns")
    times_detected: Optional[str] = Field(None, description="Timestamp of event e.g. '2:25 PM'")
    screams_panic: Optional[bool] = Field(None, description="Confirms threat severity")
    background_noise: Optional[str] = Field(None, description="Forest wind, vehicles, etc.")
    distance_estimation: Optional[str] = Field(None, description="Near / Far gunshots")

class VisualExtraction(BaseModel):
    object_detection: List[str] = Field(default_factory=list, description="Guns, people, uniforms")
    number_of_attackers: Optional[int] = Field(None, description="Count of hostile entities")
    terrain_type: Optional[str] = Field(None, description="Dense forest, meadow, etc.")
    threat_posture: Optional[str] = Field(None, description="Carrying weapons, running, crouching")
    environment_clues: List[str] = Field(default_factory=list, description="Fog, night, forest density")
    gps_exif: Optional[str] = Field(None, description="Timestamp + Location metadata")
    weather: Optional[str] = Field(None, description="Day/Night, Rain/Clear")

class TextExtraction(BaseModel):
    entities: List[str] = Field(default_factory=list, description="Attackers, civilians, weapons, vehicles")
    locations: List[str] = Field(default_factory=list, description="Baisaran, Pahalgam, forest, valley")
    events: List[str] = Field(default_factory=list, description="Firing, running, explosion")
    time_phrases: List[str] = Field(default_factory=list, description="'2:20 PM', 'afternoon'")
    sentiment_urgency: Optional[str] = Field(None, description="'Panic', 'Gunshots'")
    weapon_details: List[str] = Field(default_factory=list, description="AK-47, pistol")
    movement_direction: Optional[str] = Field(None, description="'From forest', 'towards meadow'")
    role_identification: List[str] = Field(default_factory=list, description="Attacker/Civilian")
    
    class Config:
        # Allow extra fields and be more flexible with types
        extra = 'allow'
        arbitrary_types_allowed = True

class RakshakState(BaseModel):
    """The shared state passed between nodes"""
    # Input
    file_path: str
    file_name: str
    file_type: Literal["text", "audio", "video", "image", "unknown"] = "unknown"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Extracted Data (Populated by specific nodes)
    audio_data: Optional[AudioExtraction] = None
    visual_data: Optional[VisualExtraction] = None
    text_data: Optional[TextExtraction] = None
    
    # Final Output
    processing_status: str = "pending"
    error_message: Optional[str] = None
    summary: Optional[str] = None # Brief description of the event

# --- 2. RAG MEMORY SYSTEM ---

class RakshakMemory:
    def __init__(self):
        print("   [Memory] Initializing RAG System...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(name="rakshak_intel_logs")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def store(self, text: str, metadata: dict):
        """Chunks, embeds, and stores text with metadata"""
        if not text: return
        
        chunks = self.splitter.split_text(text)
        ids = []
        embeddings = []
        docs = []
        metas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{metadata.get('source', 'unknown')}_{idx}_{uuid.uuid4().hex[:6]}"
            vector = self.embedder.encode(chunk).tolist()
            
            ids.append(chunk_id)
            embeddings.append(vector)
            docs.append(chunk)
            
            # Ensure metadata is flat and valid
            meta = metadata.copy()
            meta['chunk_index'] = idx
            meta['timestamp'] = meta.get('timestamp', datetime.now().isoformat())
            metas.append(meta)
            
        if ids:
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            print(f"   [Memory] Stored {len(ids)} chunks.")

# Initialize Memory Globally
memory = RakshakMemory()

# --- 3. NODES ---

def classifier_node(state: RakshakState) -> RakshakState:
    """Determines file type based on extension"""
    ext = os.path.splitext(state.file_name)[1].lower()
    if ext in ['.txt', '.pdf', '.csv', '.md']:
        state.file_type = "text"
    elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
        state.file_type = "audio"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        state.file_type = "video"
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        state.file_type = "image"
    else:
        state.file_type = "unknown"
    
    print(f"[{state.file_name}] Classified as: {state.file_type}")
    return state

def text_node(state: RakshakState) -> RakshakState:
    """Process Text Data using RAG"""
    print(f"[{state.file_name}] Processing Text...")
    
    try:
        # 1. Extract Text
        content = ""
        ext = os.path.splitext(state.file_path)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(state.file_path)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        elif ext in [".txt", ".md", ".log"]:
            with open(state.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif ext == ".csv":
            with open(state.file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    content += " | ".join(row) + "\n"
        
        content = content.strip()
        if not content:
            raise ValueError("No text content extracted")

        # 2. Analyze with Groq
        print("   [AI] Analyzing content...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Analyze this intelligence report.
        
        CONTENT: "{content[:3000]}"
        
        Extract the following in JSON format:
        - summary (A brief 1-2 sentence description of what is happening)
        - entities (List of people, groups, vehicles)
        - locations (List of specific places)
        - events (List of what happened)
        - time_phrases (List of when it happened)
        - sentiment_urgency (Panic, Routine, Critical)
        - weapon_details (List of types of weapons)
        - movement_direction (from/to)
        - role_identification (List of roles identified, e.g. ["Attacker", "Civilian"])
        """
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        analysis = json.loads(completion.choices[0].message.content)
        
        # 3. Store in Memory
        print("   [RAG] Storing in memory...")
        memory.store(content, {"source": state.file_name, "type": "text"})

        # 4. Populate State
        state.summary = analysis.get("summary", "No summary available.")
        
        # Handle movement_direction - convert dict to string if needed
        movement_dir = analysis.get("movement_direction", "Unknown")
        if isinstance(movement_dir, dict):
            movement_dir = f"From {movement_dir.get('from', 'Unknown')} to {movement_dir.get('to', 'Unknown')}"
        
        state.text_data = TextExtraction(
            entities=analysis.get("entities", []),
            locations=analysis.get("locations", []),
            events=analysis.get("events", []),
            time_phrases=analysis.get("time_phrases", []),
            sentiment_urgency=analysis.get("sentiment_urgency", "Unknown"),
            weapon_details=analysis.get("weapon_details", []),
            movement_direction=str(movement_dir),
            role_identification=analysis.get("role_identification", [])
        )
        
        print(f"   [Result] Urgency: {state.text_data.sentiment_urgency}")
        
    except Exception as e:
        print(f"   [Error] Text processing failed: {e}")
        state.error_message = str(e)
        state.text_data = TextExtraction(sentiment_urgency="Error")

    return state

def audio_node(state: RakshakState) -> RakshakState:
    """Process Audio Data using YAMNet and Whisper"""
    print(f"[{state.file_name}] Processing Audio...")
    
    try:
        # 1. Initialize Models (Lazy loading to avoid overhead if not needed)
        print("   [System] Loading YAMNet...")
        yamnet = tfhub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = yamnet.class_map_path().numpy()
        
        # Helper to parse class names
        class_names = []
        with open(class_map_path) as csv_file:
            reader = csv.reader(csv_file)
            next(reader) 
            for row in reader:
                class_names.append(row[2])
        
        # 2. Analyze Audio Events (YAMNet)
        print("   [YAMNet] analyzing events...")
        wav_data, sr = librosa.load(state.file_path, sr=16000, mono=True)
        scores, embeddings, spectrogram = yamnet(wav_data)
        mean_scores = np.mean(scores, axis=0)
        top_n_indices = np.argsort(mean_scores)[::-1][:5]
        
        detected_sounds = []
        has_speech = False
        gunshot_detected = False
        scream_detected = False
        
        for i in top_n_indices:
            sound = class_names[i]
            confidence = mean_scores[i]
            if confidence > 0.15:
                detected_sounds.append(f"{sound} ({confidence:.2f})")
                if "Speech" in sound or "Narration" in sound: has_speech = True
                if "Gunshot" in sound or "Explosion" in sound: gunshot_detected = True
                if "Scream" in sound: scream_detected = True
        
        # 3. Transcribe (Whisper)
        transcript = ""
        if has_speech:
            print("   [Whisper] Transcribing speech...")
            try:
                groq_client = Groq(api_key=GROQ_API_KEY)
                with open(state.file_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en", 
                        temperature=0.0
                    )
                transcript = transcription.text
                print(f"   [Whisper] Transcript: \"{transcript[:100]}...\"")
            except Exception as e:
                print(f"   [Whisper] Transcription failed: {e}")
                transcript = ""
        else:
            print("   [Whisper] No speech detected, skipping transcription")
        
        # 4. Generate Summary (LLM)
        print("   [AI] Generating Audio Summary...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Summarize this audio event in 1 sentence.
        Detected Sounds: {', '.join(detected_sounds)}
        Transcript: "{transcript}"
        """
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        state.summary = completion.choices[0].message.content.strip()

        # 5. RAG Integration
        if transcript:
            print("   [RAG] Storing transcript...")
            memory.store(transcript, {"source": state.file_name, "type": "audio_transcript"})

        # 6. Populate State
        state.audio_data = AudioExtraction(
            gunshot_classification="Detected" if gunshot_detected else "None",
            times_detected=datetime.now().strftime("%H:%M:%S"), 
            screams_panic=scream_detected,
            background_noise=", ".join(detected_sounds),
            distance_estimation="Unknown" 
        )
        
        print(f"   [Result] Sounds: {detected_sounds}")
        print(f"   [Result] Transcript: {transcript[:50]}...")
        
    except Exception as e:
        print(f"   [Error] Audio processing failed: {e}")
        state.error_message = str(e)
        state.audio_data = AudioExtraction(gunshot_classification="Error")

    return state

def video_node(state: RakshakState) -> RakshakState:
    """Process Video Data (Visual + Audio)"""
    print(f"[{state.file_name}] Processing Video...")
    
    try:
        # --- 1. VISUAL ANALYSIS (YOLO + BLIP) ---
        print("   [Visual] Scanning frames...")
        cap = cv2.VideoCapture(state.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        frame_interval = 30 # Check every 30 frames (~1 sec)
        frame_count = 0
        
        detected_objects_all = []
        max_people = 0
        weapon_detected = False
        descriptions = []
        
        # Load Models
        model_person = YOLO(PERSON_MODEL_PATH)
        # CHANGE 1: Increase weapon detection confidence to reduce false positives
        model_weapon = YOLO(WEAPON_MODEL_PATH) if os.path.exists(WEAPON_MODEL_PATH) else None
        # CHANGE 2: Use BLIP-Large for better accuracy (less hallucination)
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # YOLO Person
                p_res = model_person.predict(frame, classes=[0], verbose=False, conf=0.4)
                p_count = len(p_res[0].boxes)
                max_people = max(max_people, p_count)
                
                # YOLO Weapon
                # CHANGE 3: Increase confidence to 0.50 to reduce false positives (sticks detected as knives)
                w_names = []
                if model_weapon:
                    w_res = model_weapon.predict(frame, verbose=False, conf=0.50)
                    for box in w_res[0].boxes:
                        cls_id = int(box.cls[0])
                        w_name = w_res[0].names[cls_id]
                        w_names.append(w_name)
                        detected_objects_all.append(w_name)
                        weapon_detected = True

                # BLIP (Only if interesting)
                description = ""
                if p_count > 0 or w_names:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    blip_out = captioner(pil_img, max_new_tokens=20)
                    description = blip_out[0]['generated_text']
                    descriptions.append(description)
                
                # Log frame details if something detected
                if p_count > 0 or w_names:
                    print(f"      Time {timestamp:.1f}s | Persons: {p_count} | Weapons: {w_names} | Scene: '{description}'")
            
            frame_count += 1
        cap.release()
        
        # --- 2. AUDIO ANALYSIS (Extract -> YAMNet -> Whisper) ---
        print("   [Audio] Extracting track...")
        audio_path = os.path.join(TEMP_DIR, f"{state.file_name}.wav")
        # Use ffmpeg to extract audio
        subprocess.run(
            f'ffmpeg -y -i "{state.file_path}" -vn -ac 1 -ar 16000 "{audio_path}" -loglevel quiet', 
            shell=True
        )
        
        detected_sounds = []
        transcript = ""
        gunshot_detected = False
        scream_detected = False
        
        if os.path.exists(audio_path):
            print("   [Audio] Analyzing track...")
            # YAMNet
            yamnet = tfhub.load('https://tfhub.dev/google/yamnet/1')
            class_map_path = yamnet.class_map_path().numpy()
            class_names = []
            with open(class_map_path) as csv_file:
                reader = csv.reader(csv_file)
                next(reader) 
                for row in reader:
                    class_names.append(row[2])
            
            wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            scores, embeddings, spectrogram = yamnet(wav_data)
            mean_scores = np.mean(scores, axis=0)
            top_n_indices = np.argsort(mean_scores)[::-1][:5]
            
            has_speech = False
            for i in top_n_indices:
                sound = class_names[i]
                confidence = mean_scores[i]
                if confidence > 0.10:
                    detected_sounds.append(sound)
                    if "Speech" in sound or "Narration" in sound: has_speech = True
                    if "Gunshot" in sound or "Explosion" in sound: gunshot_detected = True
                    if "Scream" in sound: scream_detected = True
            
            # Whisper
            if has_speech:
                print("   [Audio] Transcribing speech...")
                groq_client = Groq(api_key=GROQ_API_KEY)
                with open(audio_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en", 
                        temperature=0.0
                    )
                transcript = transcription.text
            
            # Cleanup audio file
            os.remove(audio_path)
        
        # --- 3. GENERATE SUMMARY (LLM with Anti-Hallucination) ---
        print("   [AI] Generating Video Summary...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        unique_descriptions = list(set(descriptions))[:5]
        
        # CHANGE 4: Anti-Hallucination System Prompt
        system_prompt = """You are a military surveillance AI analyst.

CRITICAL RULES:
1. TRUST object detection (Weapons/Persons counts) over scene descriptions
2. Scene descriptions may hallucinate (e.g., 'bears', 'planes', 'trucks' in camouflage/foliage). IGNORE nonsensical descriptions
3. Focus ONLY on: Armed individuals, firing positions, aggressive posture, actual threats
4. If Audio indicates 'Gunshot' or 'Explosion', assume ACTIVE COMBAT even if visuals are unclear
5. If descriptions seem random/unrelated, state "Visual quality unclear" and rely on object detection
6. Do NOT mention animals, vehicles, or objects unless they are clearly relevant to security threat"""

        user_prompt = f"""
INTELLIGENCE DATA:
- Visual Scene Descriptions: {', '.join(unique_descriptions) if unique_descriptions else 'No clear descriptions'}
- CONFIRMED Objects Detected: {len(list(set(detected_objects_all)))} weapons, {max_people} persons
- Audio Events: {', '.join(detected_sounds) if detected_sounds else 'No significant sounds'}
- Audio Transcript: "{transcript if transcript else 'No speech detected'}"

TASK:
Provide a 1-2 sentence tactical summary. Prioritize object detection data over scene descriptions.
If gunshots/explosions detected in audio, emphasize active threat regardless of visual clarity."""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.3  # Lower temperature for more factual output
        )
        state.summary = completion.choices[0].message.content.strip()

        # --- 4. RAG INTEGRATION ---
        combined_text = f"Summary: {state.summary}\nTranscript: {transcript}\nVisuals: {', '.join(descriptions)}"
        if combined_text.strip():
            print("   [RAG] Storing video analysis...")
            memory.store(combined_text, {"source": state.file_name, "type": "video_analysis"})

        # --- 5. POPULATE STATE ---
        state.visual_data = VisualExtraction(
            object_detection=list(set(detected_objects_all)),
            number_of_attackers=max_people,
            threat_posture="Armed" if weapon_detected else "Unarmed",
            environment_clues=list(set(descriptions))[:5], # Top 5 unique descriptions
            weather="Unknown"
        )
        
        state.audio_data = AudioExtraction(
            gunshot_classification="Detected" if gunshot_detected else "None",
            times_detected="Multiple",
            screams_panic=scream_detected,
            background_noise=", ".join(detected_sounds),
            distance_estimation="Unknown"
        )
        
        print(f"   [Result] Visual: {max_people} people, Weapons: {list(set(detected_objects_all))}")
        print(f"   [Result] Audio: {detected_sounds}")
        
    except Exception as e:
        print(f"   [Error] Video processing failed: {e}")
        state.error_message = str(e)
        state.visual_data = VisualExtraction(object_detection=["Error"])
        state.audio_data = AudioExtraction(gunshot_classification="Error")

    return state

def image_node(state: RakshakState) -> RakshakState:
    """Process Image Data using YOLO and BLIP"""
    print(f"[{state.file_name}] Processing Image...")
    
    try:
        # 1. Load Image
        img_cv2 = cv2.imread(state.file_path)
        img_pil = Image.open(state.file_path)
        
        # 2. Object Detection (YOLO)
        print("   [YOLO] Detecting objects...")
        detected_objects = []
        person_count = 0
        weapon_count = 0
        
        # Person Detection
        model_person = YOLO(PERSON_MODEL_PATH)
        res_p = model_person.predict(img_cv2, classes=[0], verbose=False, conf=0.35)
        person_count = len(res_p[0].boxes)
        if person_count > 0:
            detected_objects.append(f"{person_count} Person(s)")
            
        # Weapon Detection
        # CHANGE 6: Increase confidence to 0.50 to reduce false positives
        if os.path.exists(WEAPON_MODEL_PATH):
            model_weapon = YOLO(WEAPON_MODEL_PATH)
            res_w = model_weapon.predict(img_cv2, verbose=False, conf=0.50)
            weapon_count = len(res_w[0].boxes)
            w_names = [res_w[0].names[int(c)] for c in res_w[0].boxes.cls]
            if w_names:
                detected_objects.extend(list(set(w_names)))
        
        # 3. Scene Captioning (BLIP-Large for better accuracy)
        print("   [BLIP] Generating caption...")
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
        caption_result = captioner(img_pil)
        description = caption_result[0]['generated_text']
        
        # CHANGE 5: Filter out obvious hallucinations
        hallucination_keywords = ['bear', 'plane', 'airplane', 'aircraft', 'truck', 'elephant', 'giraffe', 'zebra']
        if any(keyword in description.lower() for keyword in hallucination_keywords):
            print(f"   [BLIP] Warning: Possible hallucination detected: '{description}'")
            description = f"Scene unclear - Objects detected: {detected_objects}"
        
        # 4. Set Summary
        state.summary = description

        # 5. RAG Integration
        print("   [RAG] Storing image caption...")
        memory.store(description, {"source": state.file_name, "type": "image_caption"})

        # 6. Populate State
        state.visual_data = VisualExtraction(
            object_detection=detected_objects,
            number_of_attackers=person_count, # Assumption: all persons are potential threats in this context
            threat_posture="Armed" if weapon_count > 0 else "Unknown",
            environment_clues=[description],
            weather="Unknown" # Could be inferred from BLIP description
        )
        
        print(f"   [Result] Objects: {detected_objects}")
        print(f"   [Result] Description: {description}")
        
    except Exception as e:
        print(f"   [Error] Image processing failed: {e}")
        state.error_message = str(e)
        state.visual_data = VisualExtraction(object_detection=["Error"])

    return state

def save_node(state: RakshakState) -> RakshakState:
    """Saves result to JSON and moves file to processed"""
    print(f"[{state.file_name}] Saving Results...")
    
    # 1. Generate JSON Output
    output_data = state.model_dump(exclude={'file_path'}) # Don't save absolute path if not needed
    
    json_name = f"{os.path.splitext(state.file_name)[0]}_RESULT.json"
    json_path = os.path.join(RESULT_DIR, json_name)
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # 2. Move File to Processed
    dest_path = os.path.join(PROCESSED_DIR, state.file_name)
    if os.path.exists(dest_path):
        os.remove(dest_path) # Overwrite existing
    shutil.move(state.file_path, dest_path)
    
    state.processing_status = "completed"
    print(f"[{state.file_name}] Saved to {json_name} and archived.")
    return state

# --- 4. GRAPH CONSTRUCTION ---

def route_file(state: RakshakState) -> Literal["text", "audio", "video", "image", "unknown"]:
    return state.file_type

workflow = StateGraph(RakshakState)

workflow.add_node("classifier", classifier_node)
workflow.add_node("text", text_node)
workflow.add_node("audio", audio_node)
workflow.add_node("video", video_node)
workflow.add_node("image", image_node)
workflow.add_node("save", save_node)

workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_file,
    {
        "text": "text",
        "audio": "audio",
        "video": "video",
        "image": "image",
        "unknown": "save" # Unknown files just get moved/logged
    }
)

workflow.add_edge("text", "save")
workflow.add_edge("audio", "save")
workflow.add_edge("video", "save")
workflow.add_edge("image", "save")
workflow.add_edge("save", END)

app = workflow.compile()

# --- 5. EXECUTION LOOP (Async/Parallel Processing) ---

def process_single_file(filename: str) -> tuple[str, bool, str]:
    """Process a single file through the workflow"""
    file_path = os.path.join(INCOMING_DIR, filename)
    
    # Create initial state
    initial_state = RakshakState(
        file_path=file_path,
        file_name=filename
    )
    
    try:
        # Invoke Graph
        app.invoke(initial_state)
        return (filename, True, "Success")
    except Exception as e:
        return (filename, False, str(e))


def run_hybrid_system():
    """
    Hybrid Execution Strategy (CPU-Friendly):
    1. Non-Video files (Text/Audio/Image) -> Process in PARALLEL (Fast, lightweight)
    2. Video files -> Process SEQUENTIALLY (Heavy CPU/GPU usage, avoid overload)
    
    This prevents CPU spiking and memory issues when processing multiple videos.
    """
    print(">>> RAKSHAK SYSTEM STARTED (Hybrid Mode)")
    print(f">>> Monitoring: {INCOMING_DIR}")
    
    # Get all files
    all_files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not all_files:
        print(">>> No files found.")
        return
    
    # Split files into categories
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_exts]
    other_files = [f for f in all_files if os.path.splitext(f)[1].lower() not in video_exts]
    
    print(f">>> Queue: {len(video_files)} Videos (Sequential) | {len(other_files)} Others (Parallel)")
    
    results = []
    start_time = datetime.now()
    
    # --- PHASE 1: PARALLEL PROCESSING (Images, Audio, Text) ---
    if other_files:
        print(f"\n--- PHASE 1: Processing {len(other_files)} Lightweight Files in Parallel ---")
        # Use 4 workers for non-video files (they're CPU/cloud-based, not GPU-heavy)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, f): f for f in other_files}
            
            for future in as_completed(future_to_file):
                filename, success, message = future.result()
                results.append((filename, success, message))
                status = "✓" if success else "✗"
                print(f"   {status} {filename}")
    
    # --- PHASE 2: SEQUENTIAL PROCESSING (Videos) ---
    if video_files:
        print(f"\n--- PHASE 2: Processing {len(video_files)} Videos Sequentially (CPU-Safe) ---")
        for i, filename in enumerate(video_files, 1):
            print(f"   > Video {i}/{len(video_files)}: {filename}")
            
            # Process one video at a time to avoid CPU/GPU overload
            fname, success, message = process_single_file(filename)
            results.append((fname, success, message))
            
            status = "✓" if success else "✗"
            print(f"   {status} Completed")
            
            # Force garbage collection to free memory for next video
            gc.collect()
            
            # Clear GPU cache if using CUDA (torch is globally imported)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # --- SUMMARY ---
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY (Hybrid Mode)")
    print("="*60)
    print(f"Total Files: {len(all_files)}")
    print(f"  - Videos (Sequential): {len(video_files)}")
    print(f"  - Others (Parallel): {len(other_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {duration:.2f} seconds")
    if len(all_files) > 0:
        print(f"Average: {duration/len(all_files):.2f} seconds per file")
    print("="*60)


def run_system_parallel(max_workers: int = 4):
    """
    LEGACY: Full parallel processing (may cause CPU spikes with videos)
    Use run_hybrid_system() instead for better resource management
    """
    print(">>> RAKSHAK SYSTEM STARTED (Full Parallel Mode - May cause CPU spikes)")
    print(f">>> Monitoring: {INCOMING_DIR}")
    print(f">>> Max Concurrent Workers: {max_workers}")
    print(">>> WARNING: Consider using hybrid mode for better CPU management")
    
    # Get all files in incoming directory
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        print(">>> No files found.")
        return

    print(f">>> Found {len(files)} files. Starting parallel processing...\n")
    
    # Process files in parallel using ThreadPoolExecutor
    start_time = datetime.now()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_single_file, f): f for f in files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            filename, success, message = future.result()
            results.append((filename, success, message))
            
            if success:
                print(f"✓ {filename} - Completed")
            else:
                print(f"✗ {filename} - Failed: {message}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total Files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average: {duration/len(files):.2f} seconds per file")
    print("="*60)


def run_system_sequential():
    """Original sequential processing (for comparison or debugging)"""
    print(">>> RAKSHAK SYSTEM STARTED (Sequential Mode)")
    print(f">>> Monitoring: {INCOMING_DIR}")
    
    # Get all files in incoming directory
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        print(">>> No files found.")
        return

    print(f">>> Found {len(files)} files. Starting batch processing...")
    
    start_time = datetime.now()

    # Iterate through each file and run the graph
    for filename in files:
        file_path = os.path.join(INCOMING_DIR, filename)
        
        # Create initial state
        initial_state = RakshakState(
            file_path=file_path,
            file_name=filename
        )
        
        try:
            # Invoke Graph
            app.invoke(initial_state)
        except Exception as e:
            print(f"!!! Error processing {filename}: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n>>> Total Duration: {duration:.2f} seconds")


if __name__ == "__main__":
    # RECOMMENDED: Hybrid mode (parallel for lightweight, sequential for videos)
    # This prevents CPU spikes and memory issues
    run_hybrid_system()
    
    # ALTERNATIVE OPTIONS:
    # 1. Full parallel (may cause CPU spikes with multiple videos):
    #    run_system_parallel(max_workers=4)
    #
    # 2. Sequential (slowest but safest):
    #    run_system_sequential()
