import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# ===== DIAGRAM 4: MACHINE LEARNING PIPELINE =====
fig4, ax4 = plt.subplots(1, 1, figsize=(14, 10))
ax4.set_xlim(0, 14)
ax4.set_ylim(0, 10)
ax4.axis('off')

def create_box(ax, x, y, width, height, text, color, fontsize=10):
    """Create a fancy box with text"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, 
                          linewidth=2, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color='white', wrap=True)

def create_arrow(ax, x1, y1, x2, y2, label=''):
    """Create an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2.5, color='#34495E')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y + 0.2, label, fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95))

ax4.text(7, 9.6, 'ML Pipeline: Training & Inference', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# TRAINING PHASE
ax4.text(2.5, 8.8, 'TRAINING PHASE', fontsize=12, fontweight='bold', style='italic')

create_box(ax4, 2.5, 8, 2.5, 0.7, 'Raw Data\n(concepts.jsonl)', '#E74C3C', 9)
create_box(ax4, 2.5, 6.8, 2.5, 0.7, 'Text Cleaning\n& Tokenization', '#3498DB', 9)
create_arrow(ax4, 2.5, 7.65, 2.5, 7.15)

create_box(ax4, 2.5, 5.6, 2.5, 0.7, 'SentenceTransformer\nEncoding', '#9B59B6', 9)
create_arrow(ax4, 2.5, 6.45, 2.5, 5.95)

create_box(ax4, 2.5, 4.4, 2.5, 0.7, 'FAISS Index\nCreation', '#F39C12', 9)
create_arrow(ax4, 2.5, 5.25, 2.5, 4.75)

create_box(ax4, 2.5, 3.2, 2.5, 0.7, 'Save Index\n(.bin file)', '#2ECC71', 9)
create_arrow(ax4, 2.5, 4.05, 2.5, 3.55)

# INFERENCE PHASE
ax4.text(11, 8.8, 'INFERENCE PHASE', fontsize=12, fontweight='bold', style='italic')

create_box(ax4, 11, 8, 2.5, 0.7, 'User Query\n(Symptoms)', '#3498DB', 9)

create_box(ax4, 11, 6.8, 2.5, 0.7, 'normalise.py\nText Processing', '#1ABC9C', 9)
create_arrow(ax4, 11, 7.65, 11, 7.15)

create_box(ax4, 11, 5.6, 2.5, 0.7, 'Encode Query\n(Same model)', '#9B59B6', 9)
create_arrow(ax4, 11, 6.45, 11, 5.95)

create_box(ax4, 11, 4.4, 2.5, 0.7, 'FAISS Search\nTop-K retrieval', '#F39C12', 9)
create_arrow(ax4, 11, 5.25, 11, 4.75)

create_box(ax4, 11, 3.2, 2.5, 0.7, 'XLM-RoBERTa\nValidation', '#E74C3C', 9)
create_arrow(ax4, 11, 4.05, 11, 3.55)

# VALIDATION & OUTPUT
create_box(ax4, 7, 1.8, 3, 0.7, 'Confidence Filtering\n& Ranking', '#16A085', 9)
create_arrow(ax4, 4, 3.2, 5.5, 2.15)
create_arrow(ax4, 10, 3.2, 8.5, 2.15)

create_box(ax4, 7, 0.5, 3, 0.7, 'Output\n(Ranked Predictions)', '#2980B9', 9)
create_arrow(ax4, 7, 1.45, 7, 0.85)

# Add details box
details = """MODEL SPECIFICATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Embedding Model: intfloat/multilingual-e5-base
  - Multilingual support
  - 768-dimensional embeddings
  - Fine-tuned for semantic search

• Validation Model: joeddav/xlm-roberta-large-xnli
  - Cross-lingual NLI (Natural Language Inference)
  - Validates concept relevance
  - Handles ambiguity

• Search Index: FAISS FlatIP
  - Inner product similarity
  - Efficient vector search
  - Supports top-K retrieval

• Thresholds:
  - Min similarity: 0.85
  - Margin threshold: 0.15
  - Top-K retrieval: 30 concepts"""

ax4.text(0.2, 2.3, details, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FCF3CF', alpha=0.9, edgecolor='#F39C12', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/ml_pipeline_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ ML Pipeline diagram saved: ml_pipeline_diagram.png")
plt.close()

# ===== DIAGRAM 5: DISEASE CLASSIFICATION HIERARCHY =====
fig5, ax5 = plt.subplots(1, 1, figsize=(14, 11))
ax5.set_xlim(0, 14)
ax5.set_ylim(0, 11)
ax5.axis('off')

ax5.text(7, 10.7, 'Disease Classification & Medical Specialties', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Systems
systems = {
    'Cardiovascular': {'y': 9, 'diseases': ['Heart Disease', 'Hypertension', 'Arrhythmia'], 'color': '#E74C3C'},
    'Respiratory': {'y': 7.2, 'diseases': ['Asthma', 'Pneumonia', 'COPD'], 'color': '#3498DB'},
    'Gastrointestinal': {'y': 5.4, 'diseases': ['GERD', 'IBS', 'Gastritis'], 'color': '#2ECC71'},
    'Neurological': {'y': 3.6, 'diseases': ['Migraine', 'Epilepsy', 'Parkinson\'s'], 'color': '#9B59B6'},
    'Musculoskeletal': {'y': 1.8, 'diseases': ['Arthritis', 'Osteoporosis', 'Back Pain'], 'color': '#F39C12'},
}

for system, info in systems.items():
    # System header
    create_box(ax5, 2, info['y'], 2, 0.6, system, info['color'], 10)
    
    # Disease boxes
    x_start = 5.5
    for i, disease in enumerate(info['diseases']):
        x_pos = x_start + (i * 2.2)
        create_box(ax5, x_pos, info['y'], 1.8, 0.55, disease, info['color'], 8)
        create_arrow(ax5, 3.2, info['y'], x_pos - 0.9, info['y'])
    
    # Specialist recommendations
    specialist_y = info['y'] - 0.8
    specialists_text = "Recommended:\nCardiologist\nPulmonologist\nGastroenterologist\nNeurologist\nRheumatologist"
    specialist_lines = specialists_text.split('\n')[1:]
    
    for j, spec in enumerate(specialist_lines):
        if j < len(info['diseases']):
            x_pos = x_start + (j * 2.2)
            ax5.text(x_pos, specialist_y, spec, fontsize=6, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECF0F1', alpha=0.8))

# Legend
legend_text = """CLASSIFICATION LOGIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. User describes symptoms
2. Text normalized & encoded
3. FAISS finds similar disease concepts
4. XLM-RoBERTa validates relevance
5. Disease system determined
6. Specialist(s) recommended
7. Diagnostic tests suggested
8. Severity level assigned"""

ax5.text(0.5, 0.5, legend_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9, edgecolor='#16A085', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/disease_classification_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Disease Classification diagram saved: disease_classification_diagram.png")
plt.close()

# ===== DIAGRAM 6: TRIAGE & RISK ASSESSMENT =====
fig6, ax6 = plt.subplots(1, 1, figsize=(14, 10))
ax6.set_xlim(0, 14)
ax6.set_ylim(0, 10)
ax6.axis('off')

ax6.text(7, 9.6, 'Triage & Risk Assessment System', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Risk Levels
risk_levels = [
    {'level': 'CRITICAL', 'y': 8, 'color': '#8B0000', 'score': '9-10', 
     'symptoms': ['Chest pain + SOB', 'Severe bleeding', 'Loss of consciousness']},
    {'level': 'URGENT', 'y': 6, 'color': '#E74C3C', 'score': '7-8',
     'symptoms': ['Chest pain', 'Dizziness + palpitations', 'Severe headache']},
    {'level': 'NON-URGENT', 'y': 4, 'color': '#F39C12', 'score': '4-6',
     'symptoms': ['Mild fever', 'Cough', 'General discomfort']},
    {'level': 'ROUTINE', 'y': 2, 'color': '#2ECC71', 'score': '1-3',
     'symptoms': ['Common cold', 'Minor aches', 'Skin rash']},
]

for risk in risk_levels:
    # Risk level box
    create_box(ax6, 2.5, risk['y'], 2.8, 0.7, f"{risk['level']}\n(Score: {risk['score']})", risk['color'], 9)
    
    # Action box
    if risk['level'] == 'CRITICAL':
        action = "CALL 911\nERGENCY ROOM"
    elif risk['level'] == 'URGENT':
        action = "URGENT CARE\nWITHIN 1-2 hrs"
    elif risk['level'] == 'NON-URGENT':
        action = "SCHEDULE VISIT\nWITHIN 1 WEEK"
    else:
        action = "ROUTINE CHECK\nOR SELF-CARE"
    
    create_box(ax6, 6.5, risk['y'], 2.5, 0.7, action, risk['color'], 8)
    create_arrow(ax6, 4, risk['y'], 5.25, risk['y'])
    
    # Key symptoms
    symptoms_text = ' | '.join(risk['symptoms'][:2])
    ax6.text(10.5, risk['y'], symptoms_text, fontsize=7, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Triage Rules
rules_text = """TRIAGE RULES & SCORING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Symptom Detection:
   • Identify high-risk symptom combinations
   • Check for warning signs
   • Evaluate duration & severity

2. Scoring Mechanism:
   • Each symptom = points
   • Combinations = multiplier effect
   • Baseline = 1 point (any symptom)

3. Risk Factors:
   • Age (elderly/very young = higher)
   • Comorbidities
   • Symptom combination severity

4. Actions:
   • Critical → Emergency services
   • Urgent → Immediate care
   • Non-urgent → Soon
   • Routine → Self-care or appointment"""

ax6.text(0.3, 0.8, rules_text, fontsize=6.5, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.9, edgecolor='#F39C12', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/triage_system_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Triage System diagram saved: triage_system_diagram.png")
plt.close()

# ===== DIAGRAM 7: API ENDPOINTS & REQUEST-RESPONSE =====
fig7, ax7 = plt.subplots(1, 1, figsize=(14, 10))
ax7.set_xlim(0, 14)
ax7.set_ylim(0, 10)
ax7.axis('off')

ax7.text(7, 9.6, 'API Endpoints & Request/Response Flow', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Endpoints
endpoints = [
    {
        'name': 'POST /analyze',
        'y': 8,
        'method_color': '#E74C3C',
        'request': 'symptoms: str, lang?: str',
        'response': 'diseases, triage_level, specialists'
    },
    {
        'name': 'POST /predict-disease',
        'y': 6.2,
        'method_color': '#3498DB',
        'request': 'text: str, top_k?: int',
        'response': 'predictions[]'
    },
    {
        'name': 'GET /nearby-doctors',
        'y': 4.4,
        'method_color': '#2ECC71',
        'request': 'lat: float, lon: float, radius?: int',
        'response': 'doctors[]'
    },
    {
        'name': 'POST /recommend-tests',
        'y': 2.6,
        'method_color': '#9B59B6',
        'request': 'disease_label: str',
        'response': 'tests[]'
    },
]

for ep in endpoints:
    # Endpoint name
    create_box(ax7, 1.8, ep['y'], 2.2, 0.7, ep['name'], ep['method_color'], 9)
    
    # Request
    create_box(ax7, 5.5, ep['y'] + 0.4, 2.5, 0.5, f"Request:\n{ep['request']}", '#ECF0F1', 7)
    create_arrow(ax7, 3.2, ep['y'] + 0.15, 4.2, ep['y'] + 0.4)
    
    # Response
    create_box(ax7, 5.5, ep['y'] - 0.4, 2.5, 0.5, f"Response:\n{ep['response']}", '#E8F8F5', 7)
    create_arrow(ax7, 3.2, ep['y'] - 0.15, 4.2, ep['y'] - 0.4)

# Response format
response_format = """RESPONSE FORMAT (JSON):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "status": "success",
  "timestamp": "2025-12-05T10:30:00",
  "input": "user symptom text",
  "diseases": [
    {
      "label": "Disease name",
      "description": "...",
      "score": 0.92,
      "system": "System",
      "specialists": ["..."],
      "recommended_tests": ["..."]
    }
  ],
  "triage": {
    "level": "URGENT",
    "recommendation": "..."
  },
  "nearby_doctors": [
    {
      "name": "...",
      "distance": 2.5,
      "address": "..."
    }
  ]
}"""

ax7.text(8.5, 4.8, response_format, fontsize=6, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#F4ECF7', alpha=0.9, edgecolor='#9B59B6', linewidth=2))

# Server info
server_info = """SERVER CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━
Framework: FastAPI
Host: 0.0.0.0
Port: 8000
Workers: 4
CORS: Enabled (*)
Auth: None (Development)"""

ax7.text(0.3, 1.2, server_info, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FEF5E7', alpha=0.9, edgecolor='#F39C12', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/api_endpoints_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ API Endpoints diagram saved: api_endpoints_diagram.png")
plt.close()

# ===== DIAGRAM 8: DEPLOYMENT & INFRASTRUCTURE =====
fig8, ax8 = plt.subplots(1, 1, figsize=(14, 10))
ax8.set_xlim(0, 14)
ax8.set_ylim(0, 10)
ax8.axis('off')

ax8.text(7, 9.6, 'Deployment & Infrastructure Architecture', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Client Layer
create_box(ax8, 2, 8.5, 2.5, 0.7, 'Web Browser\n(Client)', '#3498DB', 9)
create_box(ax8, 6, 8.5, 2.5, 0.7, 'Mobile App\n(PWA)', '#3498DB', 9)
create_box(ax8, 10, 8.5, 2.5, 0.7, 'API Client\n(Python/cURL)', '#3498DB', 9)

# Network Layer
create_arrow(ax8, 2, 8.15, 4, 7.5)
create_arrow(ax8, 6, 8.15, 6, 7.5)
create_arrow(ax8, 10, 8.15, 8, 7.5)

create_box(ax8, 6, 7.2, 4, 0.6, 'Network/Internet', '#95A5A6', 9)

# Server Layer
create_arrow(ax8, 6, 6.9, 6, 6.3)
create_box(ax8, 6, 5.8, 5, 0.9, 'FastAPI Server (server.py)', '#E74C3C', 10)

# Core Components
create_box(ax8, 2, 4.3, 2.2, 0.7, 'Pipeline.py\nInference', '#2ECC71', 8)
create_box(ax8, 6, 4.3, 2.2, 0.7, 'FAISS Index\nSearch', '#9B59B6', 8)
create_box(ax8, 10, 4.3, 2.2, 0.7, 'Database\nConnector', '#F39C12', 8)

create_arrow(ax8, 4, 5.4, 2.8, 4.65)
create_arrow(ax8, 6, 5.4, 6, 4.65)
create_arrow(ax8, 8, 5.4, 9.2, 4.65)

# Storage Layer
create_box(ax8, 2, 2.8, 2.2, 0.7, 'Models\nDirectory', '#16A085', 8)
create_box(ax8, 6, 2.8, 2.2, 0.7, 'Data Files\n(.jsonl/.csv)', '#16A085', 8)
create_box(ax8, 10, 2.8, 2.2, 0.7, 'Cache\nStorage', '#16A085', 8)

create_arrow(ax8, 2, 3.95, 2, 3.15)
create_arrow(ax8, 6, 3.95, 6, 3.15)
create_arrow(ax8, 10, 3.95, 10, 3.15)

# Deployment options
deploy_options = """DEPLOYMENT OPTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Local Development:
   uvicorn server:app --reload

2. Production Server:
   gunicorn -w 4 server:app

3. Docker Container:
   docker build -t zsl-med .
   docker run -p 8000:8000 zsl-med

4. Cloud Platforms:
   • AWS EC2 / Lambda
   • Google Cloud Run
   • Azure Container Instances
   • Heroku

5. Load Balancing:
   • Nginx reverse proxy
   • Multiple worker processes"""

ax8.text(0.3, 1.8, deploy_options, fontsize=6.5, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FDEBD0', alpha=0.9, edgecolor='#E67E22', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/deployment_architecture_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Deployment Architecture diagram saved: deployment_architecture_diagram.png")
plt.close()

# ===== DIAGRAM 9: USER INTERACTION FLOW (UX FLOW) =====
fig9, ax9 = plt.subplots(1, 1, figsize=(14, 11))
ax9.set_xlim(0, 14)
ax9.set_ylim(0, 11)
ax9.axis('off')

ax9.text(7, 10.7, 'User Interaction & Feature Flow', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Main flow
flow_steps = [
    {'y': 9.5, 'text': '1. User Opens\nWebsite', 'color': '#3498DB'},
    {'y': 8.3, 'text': '2. Select Language\n(EN/HI/MR)', 'color': '#1ABC9C'},
    {'y': 7.1, 'text': '3. Enter Symptoms\n(Text or Voice)', 'color': '#2ECC71'},
    {'y': 5.9, 'text': '4. Click Analyze\nButton', 'color': '#F39C12'},
    {'y': 4.7, 'text': '5. API Processing\n(NLP Pipeline)', 'color': '#E74C3C'},
    {'y': 3.5, 'text': '6. View Results\n(Diseases)', 'color': '#9B59B6'},
    {'y': 2.3, 'text': '7. Request Location\nPermission', 'color': '#3498DB'},
    {'y': 1.1, 'text': '8. View Nearby\nDoctors', 'color': '#1ABC9C'},
]

for i, step in enumerate(flow_steps):
    create_box(ax9, 2.5, step['y'], 2.8, 0.8, step['text'], step['color'], 9)
    if i < len(flow_steps) - 1:
        create_arrow(ax9, 2.5, step['y'] - 0.45, 2.5, flow_steps[i+1]['y'] + 0.45)

# Features sidebar
features = """FEATURES:
━━━━━━━━━━━━━━━
✓ Multilingual Support
  EN, HI, MR

✓ Voice Input
  Browser speech-to-text

✓ Symptom Analysis
  NLP + Zero-shot learning

✓ Disease Prediction
  FAISS + XLM validation

✓ Triage Assessment
  Risk level determination

✓ Doctor Recommendations
  Specialist suggestions

✓ Nearby Doctor Finder
  OpenStreetMap integration

✓ Test Recommendations
  Diagnostic suggestions

✓ Responsive Design
  Works on all devices"""

ax9.text(5.5, 6.5, features, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FCF3CF', alpha=0.9, edgecolor='#F39C12', linewidth=2))

# Error handling
error_handling = """ERROR HANDLING:
━━━━━━━━━━━━━━━━━━━━━
• Empty input validation
• Network error recovery
• Model loading timeout
• FAISS index unavailable
• Location permission denied
• API rate limiting
• Graceful degradation
• User-friendly messages"""

ax9.text(8.5, 6.5, error_handling, fontsize=6.5, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.9, edgecolor='#E74C3C', linewidth=2))

# Performance metrics
perf_metrics = """PERFORMANCE TARGETS:
━━━━━━━━━━━━━━━━━━━
API Response Time: <2s
Frontend Load: <1s
Search Latency: <500ms
Index Queries: <100ms
Uptime Target: 99.5%
Concurrent Users: 1000+"""

ax9.text(8.5, 2.5, perf_metrics, fontsize=6.5, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#D5F4E6', alpha=0.9, edgecolor='#16A085', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/user_interaction_flow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ User Interaction Flow diagram saved: user_interaction_flow_diagram.png")
plt.close()

# ===== DIAGRAM 10: TECHNOLOGY STACK & DEPENDENCIES =====
fig10, ax10 = plt.subplots(1, 1, figsize=(14, 10))
ax10.set_xlim(0, 14)
ax10.set_ylim(0, 10)
ax10.axis('off')

ax10.text(7, 9.6, 'Technology Stack & Dependencies', 
         fontsize=16, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', alpha=0.95, edgecolor='black', linewidth=2),
         color='white')

# Stack categories
stacks = {
    'Backend': {
        'x': 1.5,
        'color': '#E74C3C',
        'techs': ['Python 3.9+', 'FastAPI', 'Uvicorn', 'Pydantic']
    },
    'ML/NLP': {
        'x': 4.5,
        'color': '#9B59B6',
        'techs': ['PyTorch', 'Transformers', 'Sentence-BERT', 'FAISS', 'XLM-RoBERTa']
    },
    'Frontend': {
        'x': 7.5,
        'color': '#3498DB',
        'techs': ['HTML5', 'JavaScript', 'CSS3', 'Web Speech API']
    },
    'Data': {
        'x': 10.5,
        'color': '#2ECC71',
        'techs': ['JSON', 'JSONL', 'CSV', 'NumPy', 'Pandas']
    },
}

for category, info in stacks.items():
    # Category header
    create_box(ax10, info['x'], 8.5, 2.2, 0.7, category, info['color'], 10)
    
    # Technology boxes
    for j, tech in enumerate(info['techs']):
        y_pos = 7.5 - (j * 0.8)
        create_box(ax10, info['x'], y_pos, 2, 0.6, tech, info['color'], 8)

# Additional tools
tools_text = """ADDITIONAL TOOLS:
━━━━━━━━━━━━━━━━━━━━
• Sentence-Transformers
• RapidFuzz (Fuzzy matching)
• Requests (HTTP client)
• CORS Middleware
• Dotenv (Configuration)
• Git (Version control)"""

ax10.text(0.5, 3.5, tools_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FCF3CF', alpha=0.9, edgecolor='#F39C12', linewidth=2))

# APIs & Services
apis_text = """EXTERNAL APIs:
━━━━━━━━━━━━━━━━━━━━
• OpenStreetMap (Doctor locator)
• Nominatim (Reverse geocoding)
• Web Speech API (Voice input)
• Geolocation API (Location)"""

ax10.text(3.8, 3.5, apis_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9, edgecolor='#16A085', linewidth=2))

# Requirements summary
reqs_text = """SYSTEM REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAM: 4GB minimum (8GB recommended)
CPU: Multi-core processor
Storage: 2GB (FAISS indices)
GPU: CUDA support (optional)
OS: Linux/Windows/macOS
Python: 3.9 or higher"""

ax10.text(7, 3.5, reqs_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.9, edgecolor='#E74C3C', linewidth=2))

# Development tools
dev_text = """DEVELOPMENT TOOLS:
━━━━━━━━━━━━━━━━━━━━━
• VS Code (IDE)
• Git (Version control)
• Pytest (Testing)
• Black (Code formatter)
• Pylint (Linter)
• Jupyter (Notebooks)"""

ax10.text(10.5, 3.5, dev_text, fontsize=7, family='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#F4ECF7', alpha=0.9, edgecolor='#9B59B6', linewidth=2))

plt.tight_layout()
plt.savefig('c:/Users/pruth/OneDrive/Desktop/My_/zsl_med_assistant/tech_stack_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Technology Stack diagram saved: tech_stack_diagram.png")
plt.close()

print("\n" + "="*60)
print("✅ ALL 10 DIAGRAMS CREATED SUCCESSFULLY!")
print("="*60)
print("\n📊 Complete Diagram Suite:")
print("   1. project_architecture_diagram.png - System architecture layers")
print("   2. data_flow_diagram.png - Data processing pipeline")
print("   3. file_structure_diagram.png - Project organization")
print("   4. ml_pipeline_diagram.png - ML training & inference")
print("   5. disease_classification_diagram.png - Medical system hierarchy")
print("   6. triage_system_diagram.png - Risk assessment & urgency levels")
print("   7. api_endpoints_diagram.png - API endpoints & requests")
print("   8. deployment_architecture_diagram.png - Infrastructure & deployment")
print("   9. user_interaction_flow_diagram.png - UX/User flows")
print("   10. tech_stack_diagram.png - Technology stack & dependencies")
print("\n" + "="*60)
print("All diagrams are 300 DPI and ready for academic submission!")
print("="*60)
