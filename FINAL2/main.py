import os
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
    gunshot_classification: Optional[str] = Field(None, description="AK-series, pistol, 7.62mm")
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
    historical_context: Optional[str] = None # Added for RAG context
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

    def retrieve(self, query: str, n_results: int = 3) -> str:
        """Retrieves relevant context"""
        if not query: return ""
        
        query_vector = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=n_results)
        
        context = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context.append(f"[Ref: {meta.get('source')}]: {doc}")
        
        return "\n\n".join(context)

# Initialize Memory Globally (Lazy load in nodes if preferred, but global is fine here)
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

        # 2. Retrieve Context (RAG)
        print("   [RAG] Retrieving context...")
        context = memory.retrieve(content[:500])
        state.historical_context = context

        # 3. Analyze with Groq
        print("   [AI] Analyzing content...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Analyze this intelligence report.
        
        CONTENT: "{content[:3000]}"
        
        HISTORICAL CONTEXT:
        {context}
        
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
        
        # 4. Store in Memory
        print("   [RAG] Storing in memory...")
        memory.store(content, {"source": state.file_name, "type": "text"})

        # 5. Populate State
        state.summary = analysis.get("summary", "No summary available.")
        state.text_data = TextExtraction(
            entities=analysis.get("entities", []),
            locations=analysis.get("locations", []),
            events=analysis.get("events", []),
            time_phrases=analysis.get("time_phrases", []),
            sentiment_urgency=analysis.get("sentiment_urgency", "Unknown"),
            weapon_details=analysis.get("weapon_details", []),
            movement_direction=analysis.get("movement_direction", "Unknown"),
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
            print("   [RAG] Retrieving context for transcript...")
            context = memory.retrieve(transcript)
            state.historical_context = context
            
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
        model_weapon = YOLO(WEAPON_MODEL_PATH) if os.path.exists(WEAPON_MODEL_PATH) else None
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                # YOLO Person
                p_res = model_person.predict(frame, classes=[0], verbose=False, conf=0.4)
                p_count = len(p_res[0].boxes)
                max_people = max(max_people, p_count)
                
                # YOLO Weapon
                w_names = []
                if model_weapon:
                    w_res = model_weapon.predict(frame, verbose=False, conf=0.4)
                    for box in w_res[0].boxes:
                        cls_id = int(box.cls[0])
                        w_name = w_res[0].names[cls_id]
                        w_names.append(w_name)
                        detected_objects_all.append(w_name)
                        weapon_detected = True

                # BLIP (Only if interesting)
                if p_count > 0 or w_names:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    blip_out = captioner(pil_img, max_new_tokens=20)
                    descriptions.append(blip_out[0]['generated_text'])
            
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
        
        # --- 3. GENERATE SUMMARY (LLM) ---
        print("   [AI] Generating Video Summary...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        unique_descriptions = list(set(descriptions))[:5]
        prompt = f"""
        Summarize this video event in 1-2 sentences.
        Visual Observations: {', '.join(unique_descriptions)}
        Detected Objects: {', '.join(list(set(detected_objects_all)))}
        Audio Sounds: {', '.join(detected_sounds)}
        Audio Transcript: "{transcript}"
        """
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        state.summary = completion.choices[0].message.content.strip()

        # --- 4. RAG INTEGRATION ---
        combined_text = f"Summary: {state.summary}\nTranscript: {transcript}\nVisuals: {', '.join(descriptions)}"
        if combined_text.strip():
            print("   [RAG] Retrieving context...")
            context = memory.retrieve(combined_text[:500])
            state.historical_context = context
            
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
        if os.path.exists(WEAPON_MODEL_PATH):
            model_weapon = YOLO(WEAPON_MODEL_PATH)
            res_w = model_weapon.predict(img_cv2, verbose=False, conf=0.30)
            weapon_count = len(res_w[0].boxes)
            w_names = [res_w[0].names[int(c)] for c in res_w[0].boxes.cls]
            if w_names:
                detected_objects.extend(list(set(w_names)))
        
        # 3. Scene Captioning (BLIP)
        print("   [BLIP] Generating caption...")
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        caption_result = captioner(img_pil)
        description = caption_result[0]['generated_text']
        
        # 4. Set Summary
        state.summary = description

        # 5. RAG Integration
        print("   [RAG] Retrieving context...")
        context = memory.retrieve(description)
        state.historical_context = context
        
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

# --- 5. EXECUTION LOOP (Handles Multiple Files) ---

def run_system():
    print(">>> RAKSHAK SYSTEM STARTED")
    print(f">>> Monitoring: {INCOMING_DIR}")
    
    # Get all files in incoming directory
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        print(">>> No files found.")
        return

    print(f">>> Found {len(files)} files. Starting batch processing...")

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

if __name__ == "__main__":
    run_system()
