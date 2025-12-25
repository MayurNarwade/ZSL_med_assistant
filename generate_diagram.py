import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
color_frontend = '#3498DB'      # Blue
color_api = '#E74C3C'           # Red
color_pipeline = '#2ECC71'      # Green
color_data = '#F39C12'          # Orange
color_models = '#9B59B6'        # Purple
color_utils = '#1ABC9C'         # Teal

def create_box(ax, x, y, width, height, text, color, fontsize=10):
    """Create a fancy box with text"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, 
                          linewidth=2, alpha=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color='white', wrap=True)

def create_arrow(ax, x1, y1, x2, y2, label=''):
    """Create an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='#34495E')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y + 0.2, label, fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ===== LAYER 1: FRONTEND =====
ax.text(0.5, 11.5, "FRONTEND LAYER", fontsize=14, fontweight='bold')
create_box(ax, 2, 10.5, 2.5, 0.8, 'index.html\n(UI/Forms)', color_frontend, 9)
create_box(ax, 5, 10.5, 2.5, 0.8, 'script.js\n(Logic/API Calls)', color_frontend, 9)
create_box(ax, 8, 10.5, 2.5, 0.8, 'style.css\n(Styling)', color_frontend, 9)

# ===== LAYER 2: API SERVER =====
ax.text(0.5, 9.7, "API SERVER LAYER", fontsize=14, fontweight='bold')
create_box(ax, 5, 9, 3.5, 0.8, 'FastAPI Server (server.py)\nCORS + Endpoints', color_api, 9)

# Frontend to API arrows
create_arrow(ax, 2, 10.1, 4.5, 9.4)
create_arrow(ax, 5, 10.1, 5, 9.4)
create_arrow(ax, 8, 10.1, 5.5, 9.4)

# ===== LAYER 3: CORE PIPELINE =====
ax.text(0.5, 8.3, "NLP PIPELINE LAYER", fontsize=14, fontweight='bold')
create_box(ax, 1.5, 7.3, 2.8, 0.8, 'pipeline.py\n(Main Inference)', color_pipeline, 9)
create_box(ax, 5, 7.3, 2.8, 0.8, 'pipeline_disease.py\n(Disease Prediction)', color_pipeline, 9)
create_box(ax, 8.5, 7.3, 2.8, 0.8, 'normalise.py\n(Text Preprocessing)', color_utils, 9)

# API to Pipeline
create_arrow(ax, 4.2, 8.6, 2.5, 7.7)
create_arrow(ax, 5, 8.6, 5, 7.7)
create_arrow(ax, 5.8, 8.6, 7.5, 7.7)

# ===== LAYER 4: SPECIALIZED MODULES =====
ax.text(0.5, 6.5, "ANALYSIS MODULES", fontsize=14, fontweight='bold')
create_box(ax, 1.5, 5.5, 2.5, 0.8, 'triage.py\n(Risk Assessment)', color_pipeline, 9)
create_box(ax, 4.5, 5.5, 2.5, 0.8, 'recommend.py\n(Specialists/Tests)', color_pipeline, 9)
create_box(ax, 7.5, 5.5, 2.5, 0.8, 'places_client.py\n(Doctor Locator)', color_pipeline, 9)

# Pipeline to Modules
create_arrow(ax, 1.5, 6.9, 1.5, 5.9)
create_arrow(ax, 3.5, 6.9, 4.5, 5.9)
create_arrow(ax, 5.5, 6.9, 7.5, 5.9)

# ===== LAYER 5: ML MODELS =====
ax.text(0.5, 4.7, "ML MODELS LAYER", fontsize=14, fontweight='bold')
create_box(ax, 2, 3.8, 2.8, 0.8, 'SentenceTransformer\n(intfloat/e5-base)', color_models, 9)
create_box(ax, 5.5, 3.8, 2.8, 0.8, 'XLM-RoBERTa-XNLI\n(NLI Classification)', color_models, 9)
create_box(ax, 9, 3.8, 2.8, 0.8, 'FAISS Index\n(Vector Search)', color_models, 9)

# Modules to Models
create_arrow(ax, 1.5, 5.1, 2, 4.2)
create_arrow(ax, 4.5, 5.1, 5.5, 4.2)
create_arrow(ax, 7.5, 5.1, 9, 4.2)

# ===== LAYER 6: DATA =====
ax.text(0.5, 3, "DATA LAYER", fontsize=14, fontweight='bold')
create_box(ax, 2, 2.1, 2.5, 0.8, 'concepts.jsonl\n(Medical Terms)', color_data, 9)
create_box(ax, 5, 2.1, 2.5, 0.8, 'diseases.jsonl\n(Disease DB)', color_data, 9)
create_box(ax, 8, 2.1, 2.5, 0.8, 'phrasebook.csv\n(Translations)', color_data, 9)

# Models to Data
create_arrow(ax, 2, 3.4, 2, 2.5)
create_arrow(ax, 5.5, 3.4, 5, 2.5)
create_arrow(ax, 9, 3.4, 8, 2.5)

# ===== LAYER 7: BUILD SCRIPTS =====
ax.text(0.5, 1.3, "BUILD & INDEXING", fontsize=14, fontweight='bold')
create_box(ax, 2.5, 0.4, 2.8, 0.7, 'build_index.py\n(FAISS Indexing)', color_data, 8)
create_box(ax, 6.5, 0.4, 2.8, 0.7, 'build_disease_index.py\n(Disease FAISS)', color_data, 8)

# Data to Build scripts
create_arrow(ax, 2, 1.7, 2.5, 0.75)
create_arrow(ax, 5, 1.7, 6.5, 0.75)

# ===== SIDE: REQUEST FLOW =====
ax.text(11.5, 11, "REQUEST FLOW", fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

flow_text = """
1. User input (symptoms)
   ↓
2. Frontend sends request
   ↓
3. API receives & routes
   ↓
4. Pipeline normalizes text
   ↓
5. SentenceTransformer encodes
   ↓
6. FAISS retrieves concepts
   ↓
7. XLM-RoBERTa validates
   ↓
8. Triage assesses urgency
   ↓
9. Recommend specialists
   ↓
10. Return results to UI
"""

ax.text(11.5, 6.5, flow_text, fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        verticalalignment='top')

# ===== TITLE =====
ax.text(8, 11.7, 'ZSL Medical Assistant - System Architecture', 
        fontsize=16, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.9, edgecolor='black', linewidth=2),
        color='white')

# ===== LEGEND =====
legend_elements = [
    mpatches.Patch(facecolor=color_frontend, edgecolor='black', label='Frontend (HTML/JS/CSS)'),
    mpatches.Patch(facecolor=color_api, edgecolor='black', label='API Server (FastAPI)'),
    mpatches.Patch(facecolor=color_pipeline, edgecolor='black', label='NLP Pipeline'),
    mpatches.Patch(facecolor=color_utils, edgecolor='black', label='Utilities'),
    mpatches.Patch(facecolor=color_models, edgecolor='black', label='ML Models'),
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Data & Indexing'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9,
         bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/project_architecture_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Architecture diagram saved: project_architecture_diagram.png")
plt.close()

# ===== CREATE A SECOND DIAGRAM: DATA FLOW =====
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(7, 9.7, 'Data Flow & Component Interaction', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.9, edgecolor='black', linewidth=2),
         color='white')

# User Input
create_box(ax2, 7, 8.8, 3, 0.6, 'User Input\n(Text/Voice)', '#3498DB', 9)

# Preprocessing
create_box(ax2, 2, 7.5, 2.5, 0.7, 'normalise.py\nText Cleaning', '#1ABC9C', 8)
create_arrow(ax2, 5.5, 8.5, 3.2, 7.9)

# Embedding
create_box(ax2, 7, 7.5, 2.8, 0.7, 'SentenceTransformer\nEmbedding', '#9B59B6', 8)
create_arrow(ax2, 7, 8.5, 7, 7.85)

# FAISS Search
create_box(ax2, 12, 7.5, 2.5, 0.7, 'FAISS Index\nSearch Top-K', '#F39C12', 8)
create_arrow(ax2, 8.4, 7.5, 10.7, 7.5)

# Retrieved Concepts
create_box(ax2, 3, 5.8, 2.5, 0.7, 'Retrieved\nConcepts', '#2ECC71', 8)
create_arrow(ax2, 2, 7.1, 2.5, 6.2)

# XLM-RoBERTa NLI
create_box(ax2, 7, 5.8, 2.8, 0.7, 'XLM-RoBERTa\nValidation/NLI', '#9B59B6', 8)
create_arrow(ax2, 7, 7.15, 7, 6.15)

# Disease Predictions
create_box(ax2, 12, 5.8, 2.5, 0.7, 'Disease\nPredictions', '#E74C3C', 8)
create_arrow(ax2, 12, 7.15, 12, 6.15)

# Triage
create_box(ax2, 2, 4.3, 2.5, 0.7, 'Triage.py\nUrgency Level', '#3498DB', 8)
create_arrow(ax2, 3, 5.45, 2.5, 4.65)

# Specialists Recommendation
create_box(ax2, 7, 4.3, 2.8, 0.7, 'recommend.py\nSpecialists & Tests', '#2ECC71', 8)
create_arrow(ax2, 7, 5.45, 7, 4.65)

# Nearby Doctors
create_box(ax2, 12, 4.3, 2.5, 0.7, 'places_client.py\nDoctor Locator', '#1ABC9C', 8)
create_arrow(ax2, 12, 5.45, 12, 4.65)

# Final Response
create_box(ax2, 7, 2.5, 4, 0.8, 'API Response\n(JSON: Diseases, Specialists, Nearby Doctors)', '#E74C3C', 8)
create_arrow(ax2, 2.5, 3.95, 5.2, 2.9)
create_arrow(ax2, 7, 3.95, 7, 2.9)
create_arrow(ax2, 12, 3.95, 8.8, 2.9)

# UI Display
create_box(ax2, 7, 1.2, 4, 0.7, 'Frontend Display\n(Results to User)', '#3498DB', 9)
create_arrow(ax2, 7, 2.1, 7, 1.55)

# Add detailed boxes on the side
detail_y = 8
details = [
    ("Key Models:", ["• Encoder: intfloat/multilingual-e5-base", "• Validator: joeddav/xlm-roberta-large-xnli", "• Index: FAISS FlatIP"], 0.7),
    ("Input Processing:", ["• Normalize text", "• Remove special chars", "• Handle multiple languages"], 3.5),
    ("Output Components:", ["• Disease predictions (top-k)", "• Urgency triage level", "• Specialist recommendations", "• Nearby doctors (OSM)"], 6.2),
]

for title, items, x_pos in details:
    ax2.text(x_pos, detail_y, title, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    for i, item in enumerate(items):
        ax2.text(x_pos, detail_y - 0.3 - (i*0.25), item, fontsize=7, family='monospace')

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/data_flow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Data flow diagram saved: data_flow_diagram.png")
plt.close()

# ===== CREATE THIRD DIAGRAM: FILE STRUCTURE =====
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 10))
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 10)
ax3.axis('off')

# Title
ax3.text(6, 9.7, 'Project File Structure & Organization', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.9, edgecolor='black', linewidth=2),
         color='white')

# Root level files
ax3.text(1, 9, 'ROOT DIRECTORY', fontsize=11, fontweight='bold')
root_files = [
    "📄 server.py (FastAPI main server)",
    "📄 build_index.py (Create FAISS index)",
    "📄 build_disease_index.py (Disease index)",
]
for i, f in enumerate(root_files):
    ax3.text(1.2, 8.6 - i*0.35, f, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F8F5', alpha=0.8))

# SRC folder
ax3.text(1, 7, 'SRC/ (Core Logic)', fontsize=11, fontweight='bold')
src_files = [
    "📦 __init__.py",
    "🔄 pipeline.py (Main NLP pipeline)",
    "🔄 pipeline_disease.py (Disease prediction)",
    "✂️  normalise.py (Text preprocessing)",
    "🚨 triage.py (Risk assessment)",
    "👨‍⚕️  recommend.py (Specialists/Tests)",
    "📍 places_client.py (Doctor locator)",
]
for i, f in enumerate(src_files):
    ax3.text(1.2, 6.6 - i*0.35, f, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FCF3CF', alpha=0.8))

# FRONTEND folder
ax3.text(6.5, 9, 'FRONTEND/', fontsize=11, fontweight='bold')
frontend_files = [
    "🌐 index.html (UI form & layout)",
    "⚙️  script.js (API calls & logic)",
    "🎨 style.css (Styling & layout)",
]
for i, f in enumerate(frontend_files):
    ax3.text(6.7, 8.6 - i*0.35, f, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8DAEF', alpha=0.8))

# DATA folder
ax3.text(6.5, 7.2, 'DATA/', fontsize=11, fontweight='bold')
data_files = [
    "📋 concepts.jsonl (Medical terms)",
    "📋 diseases.jsonl (Disease database)",
    "📋 phrasebook.csv (Translations)",
]
for i, f in enumerate(data_files):
    ax3.text(6.7, 6.8 - i*0.35, f, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FADBD8', alpha=0.8))

# MODELS folder
ax3.text(6.5, 5.4, 'MODELS/', fontsize=11, fontweight='bold')
model_files = [
    "🔍 faiss_index.bin (Concepts index)",
    "🔍 faiss_index_diseases.bin (Disease index)",
]
for i, f in enumerate(model_files):
    ax3.text(6.7, 5 - i*0.35, f, fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#D7DBDD', alpha=0.8))

# Flow description box
flow_box_text = """EXECUTION FLOW:
1. User accesses frontend (index.html)
2. Frontend makes API calls to server.py
3. server.py routes to pipeline.py
4. pipeline.py uses:
   - normalise.py for text cleaning
   - FAISS index for concept search
   - XLM-RoBERTa for validation
5. Result flows through:
   - triage.py (urgency check)
   - recommend.py (specialists)
   - places_client.py (doctors)
6. Response returned to frontend"""

ax3.text(1.2, 2.5, flow_box_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.9, edgecolor='#3498DB', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/file_structure_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ File structure diagram saved: file_structure_diagram.png")
plt.close()

print("\n✅ ALL DIAGRAMS CREATED SUCCESSFULLY!")
print("📊 Generated files:")
print("   1. project_architecture_diagram.png - Layered system architecture")
print("   2. data_flow_diagram.png - Data processing flow")
print("   3. file_structure_diagram.png - Project file organization")
