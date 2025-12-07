import os
import json
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(DATA_DIR, "result") # We now scan the raw result folders

# --- NEO4J CONNECTION ---
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

if not url:
    print("❌ Error: Neo4j credentials not found in .env")
    exit(1)

graph = Neo4jGraph(url=url, username=username, password=password)

def clear_database():
    """Wipes the DB clean (Optional - good for dev)"""
    print("   🗑️  Clearing existing Neo4j data...")
    graph.query("MATCH (n) DETACH DELETE n")

def ingest_raw_results():
    print(f"   🚀 Starting Raw Graph Ingestion from {RESULT_DIR}...")
    
    file_count = 0
    
    # Recursively walk through result/video, result/audio, etc.
    for root, dirs, files in os.walk(RESULT_DIR):
        for file in files:
            if file.endswith("_RESULT.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        process_file_data(data)
                        file_count += 1
                except Exception as e:
                    print(f"   ⚠️ Error processing {file}: {e}")

    print(f"   ✅ Ingestion Complete. Processed {file_count} files.")

def process_file_data(data):
    """
    Transforms a single JSON result into Graph Nodes & Edges
    """
    file_name = data.get('file_name')
    file_type = data.get('file_type')
    timestamp = data.get('timestamp')
    summary = data.get('summary', 'No summary')
    
    # 1. Create the FILE Node (The Anchor)
    graph.query("""
    MERGE (f:File {name: $name})
    SET f.type = $type, 
        f.timestamp = $time,
        f.summary = $summary
    """, {
        "name": file_name,
        "type": file_type,
        "time": timestamp,
        "summary": summary
    })

    # 2. Link to LOCATION (Spatial Anchor)
    meta = data.get('file_metadata', {})
    place = meta.get('place')
    if place and place != "Unknown":
        graph.query("""
        MERGE (l:Location {name: $place})
        WITH l
        MATCH (f:File {name: $name})
        MERGE (f)-[:RECORDED_AT]->(l)
        """, {"place": place, "name": file_name})

    # 3. Create OBJECT Nodes (Visual Evidence)
    if data.get('visual_data'):
        vis = data['visual_data']
        # Threat Posture
        posture = vis.get('threat_posture')
        if posture and posture != "Unknown":
            graph.query("""
            MATCH (f:File {name: $name})
            SET f.threat_posture = $posture
            """, {"name": file_name, "posture": posture})

        # Objects (Rifle, Person, Bunker)
        objects = vis.get('object_detection', [])
        for obj in objects:
            # Clean string (e.g., "3 Person(s)" -> "Person")
            # Simple heuristic cleaning
            clean_obj = obj.split(" ")[-1] if " " in obj else obj 
            clean_obj = clean_obj.replace("(s)", "").strip()
            
            graph.query("""
            MERGE (o:Object {name: $obj})
            WITH o
            MATCH (f:File {name: $name})
            MERGE (f)-[:VISUALLY_DETECTED]->(o)
            """, {"obj": clean_obj, "name": file_name})

    # 4. Create AUDIO EVENT Nodes (Audio Evidence)
    if data.get('audio_data'):
        aud = data['audio_data']
        bg_noise = aud.get('background_noise', '')
        
        # Gunshot Flag
        if aud.get('gunshot_classification') == "Detected":
            graph.query("""
            MERGE (e:Event {name: 'Gunshot'})
            SET e.type = 'Kinetic'
            WITH e
            MATCH (f:File {name: $name})
            MERGE (f)-[:AUDIO_DETECTED]->(e)
            """, {"name": file_name})
            
        # Panic Flag
        if aud.get('screams_panic'):
            graph.query("""
            MERGE (e:Event {name: 'Screams'})
            SET e.type = 'Distress'
            WITH e
            MATCH (f:File {name: $name})
            MERGE (f)-[:AUDIO_DETECTED]->(e)
            """, {"name": file_name})

    # 5. Extract ENTITIES from Text (Intel Reports)
    if data.get('text_data'):
        txt = data['text_data']
        # Handle Entity list (could be strings or dicts based on previous fixes)
        entities = txt.get('entities', [])
        
        if isinstance(entities, list):
            for ent in entities:
                ent_name = ent if isinstance(ent, str) else str(ent) # Force string
                graph.query("""
                MERGE (e:Entity {name: $ent})
                WITH e
                MATCH (f:File {name: $name})
                MERGE (f)-[:MENTIONS]->(e)
                """, {"ent": ent_name, "name": file_name})

if __name__ == "__main__":
    # clear_database() # Uncomment to reset graph every run
    ingest_raw_results()