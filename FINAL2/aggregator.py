import os
import json
import uuid
import itertools
from datetime import datetime
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(DATA_DIR, "result")
FINAL_INTEL_DIR = os.path.join(DATA_DIR, "final_intel")

os.makedirs(FINAL_INTEL_DIR, exist_ok=True)

# --- API KEY ROTATION ---
API_KEYS = []
if os.getenv("GROQ_API_KEY"): API_KEYS.append(os.getenv("GROQ_API_KEY"))
for i in range(1, 10):
    key = os.getenv(f"GROQ_API_KEY{i}")
    if key: API_KEYS.append(key)
API_KEYS = list(set([k for k in API_KEYS if k]))

if not API_KEYS:
    print("❌ [Aggregator] CRITICAL: No API Keys found.")
    exit(1)

KEY_CYCLE = itertools.cycle(API_KEYS)

def get_groq_client():
    return Groq(api_key=next(KEY_CYCLE))

# --- AGGREGATOR LOGIC ---

class IntelligenceAggregator:
    def __init__(self, risk_threshold=0.4):
        self.risk_threshold = risk_threshold
        self.evidence_log = []
        self.global_stats = {
            "total_files": 0,
            "high_risk_count": 0,
            "locations_involved": set()
        }

    def calculate_file_risk(self, data: dict) -> Tuple[float, List[str]]:
        """
        Calculates a deterministic risk score (0.0 - 1.0).
        """
        score = 0.0
        reasons = []
        
        # 1. AUDIO INDICATORS
        if data.get("audio_data"):
            aud = data["audio_data"]
            if aud.get("gunshot_classification") == "Detected":
                score += 0.7; reasons.append("Audio: Gunfire/Explosion")
            if aud.get("screams_panic") is True:
                score += 0.3; reasons.append("Audio: Distress Signals")

        # 2. VISUAL INDICATORS
        if data.get("visual_data"):
            vis = data["visual_data"]
            threat = str(vis.get("threat_posture", "")).lower()
            
            if "armed" in threat:
                score += 0.5; reasons.append("Visual: Armed Individual")
            
            detections = [str(x).lower() for x in vis.get("object_detection", [])]
            military_keywords = ["rifle", "gun", "tank", "bunker", "smoke", "fire", "weapon"]
            if any(w in d for d in detections for w in military_keywords):
                score += 0.2; reasons.append("Visual: Military Asset Detected")

        # 3. TEXT INDICATORS
        if data.get("text_data"):
            txt = data["text_data"]
            urgency = str(txt.get("sentiment_urgency", "")).lower()
            if "critical" in urgency or "high" in urgency or "panic" in urgency:
                score += 0.4; reasons.append("Text: Critical Intelligence")

        return min(1.0, score), reasons

    def ingest_data(self):
        """Recursively scans data/result/ and all subfolders"""
        print(f"   [Aggregator] Scanning recursive: {RESULT_DIR}")
        
        for root, dirs, files in os.walk(RESULT_DIR):
            for file in files:
                if file.endswith("_RESULT.json"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            risk_score, indicators = self.calculate_file_risk(data)
                            loc = data.get("file_metadata", {}).get("place", "Unknown")
                            
                            evidence = {
                                "file_name": data.get("file_name"),
                                "type": data.get("file_type"),
                                "location": loc,
                                "timestamp": data.get("timestamp"),
                                "risk_score": risk_score,
                                "indicators": indicators,
                                "raw_summary": data.get("summary", "No summary")[:300]
                            }
                            
                            self.evidence_log.append(evidence)
                            
                            self.global_stats["total_files"] += 1
                            if loc != "Unknown":
                                self.global_stats["locations_involved"].add(loc)
                            if risk_score >= self.risk_threshold:
                                self.global_stats["high_risk_count"] += 1
                                
                    except Exception as e:
                        print(f"   [Error reading {file}]: {e}")

    def generate_situation_report(self):
        """Uses LLM to write the final detailed Situation Report"""
        
        sorted_evidence = sorted(self.evidence_log, key=lambda x: x['risk_score'], reverse=True)
        global_risk = sorted_evidence[0]['risk_score'] if sorted_evidence else 0.0
        
        print(f"   [Aggregator] Analyzing {len(sorted_evidence)} files. Global Risk Score: {global_risk}")

        # Context for LLM
        context_str = f"GLOBAL METRICS: Total Files: {self.global_stats['total_files']}, High Risk: {self.global_stats['high_risk_count']}\n"
        context_str += "KEY EVIDENCE:\n"
        
        for ev in sorted_evidence[:10]:
            context_str += f"- [{ev['type'].upper()}] {ev['file_name']} (Loc: {ev['location']}) | Risk: {ev['risk_score']}\n"
            context_str += f"  Indicators: {ev['indicators']}\n"
            context_str += f"  Summary: {ev['raw_summary']}\n\n"

        # UPDATED PROMPT FOR DETAILED RECOMMENDATIONS
        prompt = f"""
        You are a Strategic Intelligence AI Advisor (RAKSHAK System). 
        Analyze the surveillance data below.
        
        DATA:
        {context_str}
        
        INSTRUCTIONS:
        1. **Situation Status:** Define the headline (e.g., "Active Ambush", "Routine Activity").
        2. **Briefing:** Synthesize what is happening across the sector.
        3. **Key Threats:** List specific verified threats.
        4. **Strategic Options (Crucial):** 
           - Do NOT give direct orders (e.g. "Shoot now"). 
           - Instead, provide a **detailed set of tactical considerations** for the Commander.
           - Include suggestions for **Containment**, **Reconnaissance**, and **Protocol**.
           - Make this section detailed (3-4 sentences) suitable for a decision-support pop-up.

        Output JSON format:
        {{
            "situation_status": "String (Headline)",
            "risk_level": "LOW / MEDIUM / HIGH / CRITICAL",
            "situation_briefing": "Detailed paragraph",
            "key_threats": ["List of specific threats"],
            "recommended_action": "Detailed paragraph suggesting tactical options, recon needs, and safety protocols."
        }}
        """

        try:
            client = get_groq_client()
            ai_response = json.loads(client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            ).choices[0].message.content)
        except Exception as e:
            print(f"   [AI Synthesis Failed]: {e}")
            ai_response = {
                "situation_status": "Analysis Failed",
                "risk_level": "UNKNOWN",
                "situation_briefing": "Could not synthesize report due to API error.",
                "key_threats": [],
                "recommended_action": "System error. Manual review of raw intelligence files is required immediately."
            }

        # Final JSON Structure
        final_output = {
            "report_id": f"SITREP_{uuid.uuid4().hex[:8]}",
            "generated_at": datetime.now().isoformat(),
            "locations_monitored": list(self.global_stats["locations_involved"]),
            "automated_risk_score": global_risk,
            "ai_analysis": ai_response,
            "evidence_ledger": sorted_evidence
        }

        # Save
        out_path = os.path.join(FINAL_INTEL_DIR, "GLOBAL_SITUATION_REPORT.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
            
        print(f"\n✅ REPORT GENERATED: {out_path}")
        print(f"   Status: {ai_response.get('situation_status')}")
        print(f"   Risk: {ai_response.get('risk_level')}")

if __name__ == "__main__":
    print("\n" + "="*60 + "\n>>> STARTING INTELLIGENCE AGGREGATOR\n" + "="*60)
    agg = IntelligenceAggregator(risk_threshold=0.4)
    agg.ingest_data()
    
    if agg.evidence_log:
        agg.generate_situation_report()
    else:
        print("   No intelligence files found to analyze.")
    print("="*60)