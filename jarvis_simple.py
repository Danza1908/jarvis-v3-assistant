"""
JARVIS V3 - Complete AI Assistant with Voice Interface
=====================================================
Gemini Pro as the main brain commanding specialized agents
with full voice interaction capabilities.
"""

import streamlit as st
import os
from datetime import datetime, timedelta
import json
import base64
import hashlib
import secrets
import re
import random
import requests
from urllib.parse import quote

# Try to import required packages
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Check if running in cloud environment
CLOUD_ENVIRONMENT = (
    os.getenv('STREAMLIT_SHARING_MODE') == '1' or  # Streamlit Cloud
    os.getenv('DYNO') is not None or  # Heroku
    os.getenv('K_SERVICE') is not None or  # Google Cloud Run
    os.getenv('AWS_EXECUTION_ENV') is not None or  # AWS
    os.getenv('RAILWAY_ENVIRONMENT') is not None  # Railway
)

# Voice interface imports - graceful degradation for cloud
try:
    if not CLOUD_ENVIRONMENT:  # Only try to import voice on local
        import speech_recognition as sr
        import pyttsx3
        VOICE_AVAILABLE = True
    else:
        VOICE_AVAILABLE = False
except ImportError:
    VOICE_AVAILABLE = False

# Audio processing - local only
try:
    if not CLOUD_ENVIRONMENT:
        import sounddevice as sd
        import numpy as np
        import scipy.io.wavfile as wav
        AUDIO_AVAILABLE = True
    else:
        AUDIO_AVAILABLE = False
except ImportError:
    AUDIO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="JARVIS V3 - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}

.user-message {
    background-color: #f0f2f6;
    border-left-color: #667eea;
}

.jarvis-message {
    background-color: #e8f4fd;
    border-left-color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Security Functions
def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key):
    """Hash API key for secure storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def validate_input(user_input):
    """Validate user input for security threats"""
    if not user_input or len(user_input.strip()) == 0:
        return False, "Empty input not allowed"
    
    # Check for potential injection attempts
    dangerous_patterns = [
        r'<script.*?>', r'javascript:', r'eval\(', r'exec\(',
        r'__import__', r'subprocess', r'os\.system',
        r'DROP\s+TABLE', r'DELETE\s+FROM', r'UPDATE.*SET'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False, "Potentially unsafe input detected"
    
    # Check input length
    if len(user_input) > 10000:
        return False, "Input too long (max 10,000 characters)"
    
    return True, "Input validated"

def log_security_event(event_type, details):
    """Log security events"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details,
        "session_id": st.session_state.get('session_id', 'unknown')
    }
    
    # In production, this would go to a proper logging system
    if 'security_logs' not in st.session_state:
        st.session_state.security_logs = []
    
    st.session_state.security_logs.append(log_entry)
    
    # Keep only last 100 logs to prevent memory bloat
    if len(st.session_state.security_logs) > 100:
        st.session_state.security_logs = st.session_state.security_logs[-100:]

def check_rate_limiting():
    """Basic rate limiting to prevent abuse"""
    current_time = datetime.now()
    
    if 'last_requests' not in st.session_state:
        st.session_state.last_requests = []
    
    # Remove requests older than 1 minute
    st.session_state.last_requests = [
        req_time for req_time in st.session_state.last_requests 
        if current_time - req_time < timedelta(minutes=1)
    ]
    
    # Check if user has made too many requests
    if len(st.session_state.last_requests) >= 30:  # Max 30 requests per minute
        return False, "Rate limit exceeded. Please wait a moment."
    
    # Add current request
    st.session_state.last_requests.append(current_time)
    return True, "Rate limit OK"

def sanitize_output(text):
    """Sanitize output to prevent any potential issues"""
    if not text:
        return ""
    
    # Remove potential HTML/JS injection
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    return text

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'gemini_initialized' not in st.session_state:
    st.session_state.gemini_initialized = False
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = "General Assistant"
if 'session_id' not in st.session_state:
    st.session_state.session_id = generate_session_token()
if 'security_logs' not in st.session_state:
    st.session_state.security_logs = []
if 'last_requests' not in st.session_state:
    st.session_state.last_requests = []
if 'saved_api_key' not in st.session_state:
    st.session_state.saved_api_key = ""
if 'api_key_saved' not in st.session_state:
    st.session_state.api_key_saved = False

def save_api_key_locally(api_key):
    """Save API key to session state for persistence during the session"""
    if api_key and api_key.startswith('AIza') and len(api_key) > 20:
        st.session_state.saved_api_key = api_key
        st.session_state.api_key_saved = True
        # Also save to environment for this session
        os.environ["GEMINI_API_KEY"] = api_key
        return True
    return False

def get_saved_api_key():
    """Get saved API key from multiple sources (session, environment, Streamlit secrets)"""
    # Check session state first
    if st.session_state.saved_api_key:
        return st.session_state.saved_api_key
    
    # Check Streamlit secrets (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets.general:
            api_key = st.secrets.general.GEMINI_API_KEY
            if api_key and api_key != "your-gemini-api-key-here":
                return api_key
    except Exception:
        pass
    
    # Check environment variable
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key and env_key != "your-gemini-api-key-here":
        return env_key
    
    return ""

# Define specialized agents that Gemini Pro can embody
JARVIS_AGENTS = {
    "General Assistant": {
        "icon": "🤖",
        "description": "General AI assistant for any task",
        "personality": "I'm JARVIS, your general AI assistant. I can help with any questions or tasks you have."
    },
    "Project Manager": {
        "icon": "📊", 
        "description": "Project planning, timelines, and organization",
        "personality": "I'm your Project Manager. I excel at planning, organizing tasks, creating timelines, and managing projects efficiently."
    },
    "Content Creator": {
        "icon": "✍️",
        "description": "Writing, articles, blogs, and creative content", 
        "personality": "I'm your Content Creator. I specialize in writing engaging content, articles, blogs, scripts, and all forms of creative writing."
    },
    "Financial Analyst": {
        "icon": "💰",
        "description": "Budget analysis, investments, and financial planning",
        "personality": "I'm your Financial Analyst. I help with budgeting, investment analysis, financial planning, and economic insights."
    },
    "Tech Specialist": {
        "icon": "💻",
        "description": "Programming, technology, and technical solutions",
        "personality": "I'm your Tech Specialist. I can help with programming, technology solutions, debugging, and technical explanations."
    },
    "Creative Designer": {
        "icon": "🎨", 
        "description": "Design concepts, visual ideas, and creative solutions",
        "personality": "I'm your Creative Designer. I help with design concepts, visual ideas, branding, and creative problem-solving."
    },
    "Business Strategist": {
        "icon": "📈",
        "description": "Business strategy, growth planning, and market analysis", 
        "personality": "I'm your Business Strategist. I focus on business growth, strategic planning, market analysis, and competitive strategies."
    },
    "Research Assistant": {
        "icon": "🔍",
        "description": "Research, analysis, and information gathering",
        "personality": "I'm your Research Assistant. I excel at gathering information, conducting analysis, and providing comprehensive research."
    },
    "Wellness Coach": {
        "icon": "🧘",
        "description": "Health, fitness, wellness, and lifestyle advice",
        "personality": "I'm your Wellness Coach. I help with health advice, fitness planning, wellness strategies, and lifestyle improvements."
    },
    "Security Expert": {
        "icon": "🛡️",
        "description": "Cybersecurity, privacy protection, and digital safety",
        "personality": """I'm your Security Expert and Digital Guardian. I specialize in:

🔒 **Cybersecurity Protection:**
- Threat analysis and vulnerability assessment
- Malware detection and prevention strategies
- Network security and firewall configuration
- Password security and authentication methods
- Data encryption and secure communications

🛡️ **Privacy Protection:**
- Personal data protection strategies
- GDPR and privacy compliance guidance
- Social media privacy settings optimization
- Digital footprint management
- Anonymous browsing and VPN recommendations

🚨 **Threat Intelligence:**
- Latest cybersecurity threats and trends
- Zero-day vulnerability awareness
- Phishing and social engineering prevention
- Ransomware protection strategies
- IoT device security hardening

🔐 **Digital Safety Best Practices:**
- Secure software development practices
- Cloud security configuration
- Mobile device security
- Email security and encrypted communications
- Backup and disaster recovery planning

I stay updated with the latest security frameworks, NIST guidelines, OWASP standards, and emerging threats. I can help secure your digital life, business infrastructure, and provide incident response guidance."""
    }
}

# Dashboard Components
def get_jarvis_avatar_state():
    """Get JARVIS avatar based on current state"""
    if st.session_state.get('gemini_initialized', False):
        current_agent = st.session_state.get('current_agent', 'General Assistant')
        agent_icon = JARVIS_AGENTS.get(current_agent, {}).get('icon', '🤖')
        
        # Animated states
        states = {
            'idle': f"{agent_icon} 😊",
            'thinking': f"{agent_icon} 🤔", 
            'active': f"{agent_icon} ⚡",
            'secure': f"{agent_icon} 🛡️"
        }
        
        # Determine state based on recent activity
        if st.session_state.get('last_requests', []):
            return states['active']
        elif current_agent == 'Security Expert':
            return states['secure']
        else:
            return states['idle']
    else:
        return "🤖 😴"

def create_daily_agenda():
    """Create daily agenda widget"""
    st.markdown("### 📅 Daily Agenda")
    
    today = datetime.now()
    
    # Sample agenda items (in real app, these would come from a calendar API)
    if 'agenda_items' not in st.session_state:
        st.session_state.agenda_items = [
            {"time": "09:00", "task": "Team standup meeting", "type": "meeting"},
            {"time": "10:30", "task": "Review project proposals", "type": "work"},
            {"time": "12:00", "task": "Lunch break", "type": "personal"},
            {"time": "14:00", "task": "Client presentation", "type": "meeting"},
            {"time": "16:00", "task": "Code review session", "type": "work"},
            {"time": "18:00", "task": "Gym workout", "type": "health"}
        ]
    
    # Display agenda
    for item in st.session_state.agenda_items:
        icon = {"meeting": "🤝", "work": "💼", "personal": "🏠", "health": "💪"}.get(item['type'], "📋")
        st.markdown(f"**{item['time']}** {icon} {item['task']}")
    
    # Add new agenda item
    with st.expander("➕ Add New Task"):
        col1, col2 = st.columns(2)
        with col1:
            new_time = st.text_input("Time (HH:MM)", placeholder="14:30")
        with col2:
            task_type = st.selectbox("Type", ["meeting", "work", "personal", "health"])
        
        new_task = st.text_input("Task Description", placeholder="Enter task...")
        
        if st.button("Add Task", use_container_width=True):
            if new_time and new_task:
                st.session_state.agenda_items.append({
                    "time": new_time,
                    "task": new_task,
                    "type": task_type
                })
                st.session_state.agenda_items.sort(key=lambda x: x['time'])
                st.rerun()

def create_health_dashboard():
    """Create health information panel"""
    st.markdown("### 🏥 Health Dashboard")
    
    # Initialize health data
    if 'health_data' not in st.session_state:
        st.session_state.health_data = {
            'water_intake': 0,
            'steps': 0,
            'sleep_hours': 0,
            'mood': 'Good'
        }
    
    # Health metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💧 Water Intake", f"{st.session_state.health_data['water_intake']} glasses", 
                 delta=f"{8 - st.session_state.health_data['water_intake']} to goal")
        
        st.metric("🚶 Steps Today", f"{st.session_state.health_data['steps']:,}", 
                 delta=f"{10000 - st.session_state.health_data['steps']:,} to goal")
    
    with col2:
        st.metric("😴 Sleep Last Night", f"{st.session_state.health_data['sleep_hours']}h", 
                 delta=f"{8 - st.session_state.health_data['sleep_hours']}h to optimal")
        
        mood_emoji = {"Excellent": "😄", "Good": "😊", "Okay": "😐", "Poor": "😢"}
        st.metric("😊 Mood", st.session_state.health_data['mood'], 
                 delta=mood_emoji.get(st.session_state.health_data['mood'], "😊"))
    
    # Quick health actions
    st.markdown("**Quick Actions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💧 Log Water", use_container_width=True):
            st.session_state.health_data['water_intake'] += 1
            st.rerun()
    
    with col2:
        if st.button("🚶 Add Steps", use_container_width=True):
            st.session_state.health_data['steps'] += 1000
            st.rerun()
    
    with col3:
        if st.button("😊 Update Mood", use_container_width=True):
            moods = ["Excellent", "Good", "Okay", "Poor"]
            current_idx = moods.index(st.session_state.health_data['mood'])
            st.session_state.health_data['mood'] = moods[(current_idx + 1) % len(moods)]
            st.rerun()
    
    # Health tips
    health_tips = [
        "💡 Drink water every hour to stay hydrated",
        "💡 Take a 5-minute walk break every hour",
        "💡 Practice deep breathing for stress relief",
        "💡 Maintain good posture while working",
        "💡 Get 7-9 hours of sleep for optimal health"
    ]
    
    st.info(random.choice(health_tips))

def create_news_feed():
    """Create custom news feed"""
    st.markdown("### 📰 Latest Updates")
    
    # Sample news items (in real app, these would come from news APIs)
    if 'news_items' not in st.session_state:
        st.session_state.news_items = [
            {
                "title": "AI Revolution: New Breakthroughs in Machine Learning",
                "source": "Tech Today",
                "time": "2 hours ago",
                "category": "Technology",
                "summary": "Latest advancements in AI technology show promising results..."
            },
            {
                "title": "Global Markets Show Strong Performance",
                "source": "Business Weekly", 
                "time": "4 hours ago",
                "category": "Business",
                "summary": "Stock markets worldwide demonstrate resilience amid economic changes..."
            },
            {
                "title": "New Health Study Reveals Benefits of Regular Exercise",
                "source": "Health News",
                "time": "6 hours ago", 
                "category": "Health",
                "summary": "Recent research confirms the importance of daily physical activity..."
            },
            {
                "title": "Cybersecurity Alert: New Threats Emerging",
                "source": "Security Today",
                "time": "8 hours ago",
                "category": "Security", 
                "summary": "Security experts warn about new malware targeting businesses..."
            }
        ]
    
    # News categories filter
    categories = ["All", "Technology", "Business", "Health", "Security"]
    selected_category = st.selectbox("Filter by category:", categories)
    
    # Display news items
    filtered_news = st.session_state.news_items
    if selected_category != "All":
        filtered_news = [item for item in st.session_state.news_items if item['category'] == selected_category]
    
    for item in filtered_news:
        with st.expander(f"📰 {item['title']} - {item['time']}"):
            st.markdown(f"**Source:** {item['source']} | **Category:** {item['category']}")
            st.write(item['summary'])
            
            # Ask JARVIS about this news
            if st.button(f"🤖 Ask JARVIS about this", key=f"news_{item['title'][:20]}"):
                news_question = f"Tell me more about: {item['title']}"
                st.session_state.messages.append({"role": "user", "content": news_question})
                
                # Auto-select relevant agent
                if item['category'] == "Technology":
                    agent = "Tech Specialist"
                elif item['category'] == "Business":
                    agent = "Business Strategist"
                elif item['category'] == "Health":
                    agent = "Wellness Coach"
                elif item['category'] == "Security":
                    agent = "Security Expert"
                else:
                    agent = "Research Assistant"
                
                st.session_state.current_agent = agent
                st.rerun()

def get_agent_prompt(agent_name, user_message):
    """Create a specialized prompt based on the selected agent"""
    agent = JARVIS_AGENTS.get(agent_name, JARVIS_AGENTS["General Assistant"])
    
    prompt = f"""You are JARVIS, acting as a {agent_name}. 

{agent['personality']}

User message: {user_message}

Respond as this specialized agent, using your expertise in this domain. Be helpful, knowledgeable, and maintain the JARVIS personality while focusing on your specialization.

If the user's request doesn't match your specialization, acknowledge it and either:
1. Provide what help you can from your perspective, or 
2. Suggest which JARVIS agent would be better suited for their request.

Available JARVIS agents: {', '.join(JARVIS_AGENTS.keys())}
"""
    return prompt

def initialize_voice():
    """Initialize voice recognition and text-to-speech"""
    try:
        if VOICE_AVAILABLE:
            # Initialize text-to-speech
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Try to set a more natural voice
            for voice in voices:
                if 'english' in voice.name.lower() and 'male' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            engine.setProperty('rate', 180)  # Speech rate
            engine.setProperty('volume', 0.8)  # Volume level
            
            return engine
        return None
    except Exception as e:
        st.error(f"Voice initialization failed: {str(e)}")
        return None

def speak_text(text, engine):
    """Convert text to speech"""
    try:
        if engine and st.session_state.get('voice_enabled', False):
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"Speech failed: {str(e)}")

def listen_for_speech():
    """Listen for speech input"""
    try:
        if not VOICE_AVAILABLE:
            return None
            
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        with microphone as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        st.info("🔄 Processing speech...")
        text = recognizer.recognize_google(audio)
        return text
        
    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("🤷 Could not understand speech. Please try again.")
        return None
    except Exception as e:
        st.error(f"Speech recognition failed: {str(e)}")
        return None

def initialize_gemini(api_key):
    """Initialize Gemini Pro with API key"""
    try:
        if GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            
            # Use the correct model name for Gemini Pro
            # Try different model names that are available
            model_names = [
                'gemini-1.5-flash',  # Latest model
                'gemini-1.5-pro',    # Pro model
                'gemini-pro',        # Legacy name
                'models/gemini-pro'  # Full path
            ]
            
            model = None
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    # Test the connection with a simple prompt
                    test_response = model.generate_content("Hello, this is a test. Please respond with 'JARVIS is ready!'")
                    
                    if test_response and test_response.text:
                        st.session_state.gemini_model = model
                        st.session_state.gemini_initialized = True
                        st.session_state.model_name = model_name
                        st.success(f"✅ Connected using model: {model_name}")
                        return True
                except Exception as e:
                    st.warning(f"Model {model_name} failed: {str(e)}")
                    continue
            
            st.error("❌ All model attempts failed. Please check your API key and try again.")
            return False
        else:
            st.error("❌ Google Generative AI library not available")
            return False
    except Exception as e:
        st.error(f"Gemini initialization failed: {str(e)}")
        return False

def get_gemini_response(prompt, agent_name="General Assistant"):
    """Get response from Gemini Pro acting as a specialized agent"""
    try:
        if not st.session_state.get('gemini_initialized', False):
            return "❌ JARVIS is not initialized. Please activate with your API key first."
        
        model = st.session_state.get('gemini_model')
        if not model:
            return "❌ No model available. Please reconnect to JARVIS."
            
        # Create specialized prompt
        specialized_prompt = get_agent_prompt(agent_name, prompt)
        
        response = model.generate_content(specialized_prompt)
        
        if response and response.text:
            return response.text
        else:
            return "❌ No response generated. Please try again."
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            model_name = st.session_state.get('model_name', 'unknown')
            return f"❌ Model not found error. Current model: {model_name}. Please try reconnecting with your API key."
        elif "quota" in error_msg.lower():
            return "❌ API quota exceeded. Please check your Gemini API usage limits."
        elif "invalid" in error_msg.lower():
            return "❌ Invalid API key. Please check your Gemini API key."
        else:
            return f"❌ Error getting response: {error_msg}"

def auto_select_agent(user_input):
    """Automatically select the best agent based on user input"""
    user_input_lower = user_input.lower()
    
    # Keywords for each agent
    agent_keywords = {
        "Project Manager": ["project", "plan", "timeline", "schedule", "organize", "manage", "task", "deadline"],
        "Content Creator": ["write", "article", "blog", "content", "story", "script", "creative writing"],
        "Financial Analyst": ["money", "budget", "finance", "investment", "cost", "revenue", "financial"],
        "Tech Specialist": ["code", "programming", "software", "technical", "debug", "technology", "computer"],
        "Creative Designer": ["design", "visual", "graphics", "logo", "branding", "creative", "artistic"],
        "Business Strategist": ["business", "strategy", "market", "growth", "competition", "strategic"],
        "Research Assistant": ["research", "analyze", "study", "investigate", "information", "data"],
        "Wellness Coach": ["health", "fitness", "wellness", "exercise", "nutrition", "mental health"],
        "Security Expert": ["security", "cybersecurity", "privacy", "protection", "hack", "malware", "virus", 
                           "firewall", "encryption", "password", "breach", "vulnerability", "threat", "phishing",
                           "ransomware", "vpn", "authentication", "gdpr", "compliance", "backup", "antivirus"]
    }
    
    # Score each agent
    agent_scores = {}
    for agent, keywords in agent_keywords.items():
        score = sum(1 for keyword in keywords if keyword in user_input_lower)
        if score > 0:
            agent_scores[agent] = score
    
    # Return the agent with the highest score, or General Assistant if no match
    if agent_scores:
        return max(agent_scores.keys(), key=lambda x: agent_scores[x])
    else:
        return "General Assistant"

def main():
    # Initialize voice engine
    voice_engine = None
    if VOICE_AVAILABLE:
        voice_engine = initialize_voice()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 JARVIS V3 - Complete AI Assistant</h1>
        <p>Powered by Gemini Pro with Voice Interface & Specialized Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ JARVIS Control Center")
        
        # Gemini API Key input
        st.markdown("**🔑 Gemini Pro Configuration**")
        
        # Load saved key if available
        saved_key = get_saved_api_key()
        
        # Show if key is saved
        if saved_key:
            st.success("✅ API Key is saved and will be remembered!")
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🔄 Use Different Key"):
                    st.session_state.saved_api_key = ""
                    st.session_state.api_key_saved = False
                    st.session_state.gemini_initialized = False
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear"):
                    st.session_state.saved_api_key = ""
                    st.session_state.api_key_saved = False
                    st.session_state.gemini_initialized = False
                    if "GEMINI_API_KEY" in os.environ:
                        del os.environ["GEMINI_API_KEY"]
                    st.rerun()
        
        # API Key input (only show if no saved key or user wants to change)
        if not saved_key or not st.session_state.api_key_saved:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get your free API key: https://makersuite.google.com/app/apikey",
                placeholder="AIza...",
                value=""
            )
            
            if gemini_key:
                if gemini_key.startswith('AIza') and len(gemini_key) > 20:
                    # Save the key
                    if save_api_key_locally(gemini_key):
                        st.success("✅ API Key saved!")
                        
                    # Initialize Gemini
                    if not st.session_state.gemini_initialized:
                        with st.spinner("🔧 Initializing Gemini Pro..."):
                            if initialize_gemini(gemini_key):
                                st.success("✅ Gemini Pro connected!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to connect to Gemini Pro")
                    else:
                        st.success("✅ Gemini Pro is connected!")
                else:
                    st.error("❌ Invalid API key format. Should start with 'AIza'")
        else:
            # Auto-initialize with saved key
            if not st.session_state.gemini_initialized:
                with st.spinner("🔧 Connecting with saved API key..."):
                    if initialize_gemini(saved_key):
                        st.success("✅ Gemini Pro connected with saved key!")
                    else:
                        st.error("❌ Failed to connect with saved key")
                        st.session_state.saved_api_key = ""
                        st.session_state.api_key_saved = False
        
        # Agent Selection
        if st.session_state.gemini_initialized:
            st.markdown("---")
            st.markdown("**🎯 Specialized Agents**")
            
            # Auto-select toggle
            auto_select = st.checkbox("🤖 Auto-select best agent", value=True)
            
            if not auto_select:
                # Manual agent selection
                agent_options = list(JARVIS_AGENTS.keys())
                selected_agent = st.selectbox(
                    "Choose Agent",
                    agent_options,
                    index=agent_options.index(st.session_state.current_agent)
                )
                st.session_state.current_agent = selected_agent
                
                # Show agent info
                agent_info = JARVIS_AGENTS[selected_agent]
                st.info(f"{agent_info['icon']} **{selected_agent}**\n\n{agent_info['description']}")
        
        # Voice Controls
        st.markdown("---")
        st.markdown("**🎤 Voice Interface**")
        
        if VOICE_AVAILABLE:
            st.session_state.voice_enabled = st.checkbox("🔊 Enable voice responses", value=st.session_state.voice_enabled)
            
            if st.session_state.gemini_initialized:
                if st.button("🎤 Voice Input"):
                    with st.spinner("🎧 Listening for your command..."):
                        speech_text = listen_for_speech()
                        if speech_text:
                            st.session_state.voice_input = speech_text
                            st.success(f"🎤 Heard: '{speech_text}'")
        else:
            st.warning("⚠️ Voice features require: pip install speechrecognition pyttsx3")
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Status
        st.markdown("---")
        st.markdown("**📊 System Status**")
        
        if GEMINI_AVAILABLE:
            st.success("✅ Gemini Pro Available")
        else:
            st.warning("⚠️ Google Generative AI not installed")
            
        if st.session_state.gemini_initialized:
            st.success("✅ JARVIS Online")
            st.info(f"🎯 Current Agent: {st.session_state.current_agent}")
            if st.session_state.api_key_saved:
                st.success("💾 API Key Saved - No need to re-enter!")
        else:
            st.info("🔧 Add Gemini API key to activate")
            
        if VOICE_AVAILABLE:
            st.success("✅ Voice Interface Available")
        else:
            st.warning("⚠️ Voice features not available")
        
        # Security Dashboard
        st.markdown("---")
        st.markdown("**🛡️ Security Dashboard**")
        
        # Session security info
        st.success("🔒 Session Secured")
        session_id_short = st.session_state.session_id[:8] + "..."
        st.caption(f"Session ID: {session_id_short}")
        
        # Request count
        request_count = len(st.session_state.last_requests)
        if request_count == 0:
            st.success("📊 No recent requests")
        elif request_count < 15:
            st.info(f"📊 {request_count}/30 requests this minute")
        elif request_count < 25:
            st.warning(f"⚠️ {request_count}/30 requests this minute")
        else:
            st.error(f"🚨 {request_count}/30 requests - Near limit!")
        
        # Security logs summary
        if st.session_state.security_logs:
            recent_events = len([log for log in st.session_state.security_logs 
                               if datetime.fromisoformat(log['timestamp']) > datetime.now() - timedelta(minutes=5)])
            st.info(f"📋 {recent_events} security events (5 min)")
            
            # Show expandable security log
            with st.expander("🔍 View Security Logs"):
                for log in st.session_state.security_logs[-5:]:  # Show last 5 events
                    event_time = datetime.fromisoformat(log['timestamp']).strftime("%H:%M:%S")
                    st.caption(f"**{event_time}** - {log['event_type']}")
        else:
            st.success("✅ No security events")
        
        # Security Agent quick access
        if st.button("🛡️ Consult Security Expert", use_container_width=True):
            st.session_state.current_agent = "Security Expert"
            st.rerun()
        
        # Environment Status
        st.markdown("---")
        st.markdown("**🌐 Environment Status**")
        
        if CLOUD_ENVIRONMENT:
            st.info("☁️ Running in Cloud Mode")
            st.caption("Voice features disabled for cloud compatibility")
        else:
            st.success("🖥️ Running Locally")
            if VOICE_AVAILABLE:
                st.success("✅ Voice Interface Available")
            else:
                st.warning("⚠️ Voice features not available")
        
        # Feature availability
        with st.expander("📋 Feature Availability"):
            st.write("**✅ Always Available:**")
            st.write("• 🤖 All 10 AI Agents")
            st.write("• 🧠 Gemini Pro Chat")
            st.write("• 🛡️ Security Features")
            st.write("• 📊 Security Dashboard")
            st.write("• 🔐 API Key Persistence")
            
            if CLOUD_ENVIRONMENT:
                st.write("\n**⚠️ Cloud Limitations:**")
                st.write("• 🎤 Voice Interface (disabled)")
                st.write("• 📁 Local File Access (disabled)")
                st.write("• 💾 Local Storage (session only)")
            else:
                st.write("\n**🎯 Local Extras:**")
                if VOICE_AVAILABLE:
                    st.write("• 🎤 Voice Interface")
                if AUDIO_AVAILABLE:
                    st.write("• 🔊 Advanced Audio")
    
    # Main Dashboard Interface
    if st.session_state.gemini_initialized:
        # JARVIS Header with Avatar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            avatar_state = get_jarvis_avatar_state()
            st.markdown(f"""
            <div style="text-align: center; font-size: 3rem; margin: 1rem 0;">
                {avatar_state}
            </div>
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2>JARVIS Personal Assistant</h2>
                <p style="color: #666;">Your AI-powered daily companion</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dashboard Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📅 Agenda", "🏥 Health", "📰 News", "🤖 Agents"])
        
        with tab1:
            # Main chat interface
            st.header("💬 Chat with JARVIS")
            
            # Show current agent
            current_agent = st.session_state.current_agent
            agent_info = JARVIS_AGENTS[current_agent]
            
            st.info(f"""
            🎯 **Active Agent: {agent_info['icon']} {current_agent}**
            
            {agent_info['description']}
            
            JARVIS will automatically switch agents based on your requests, or you can manually select one in the sidebar.
            """)
            
            # Welcome message if no chat history
            if not st.session_state.messages:
                st.success("""
                🎉 **JARVIS V3 is online and ready!**
                
                **Try asking:**
                - "Help me plan my project timeline" (Project Manager)
                - "Write a blog post about AI" (Content Creator)
                - "Analyze my investment portfolio" (Financial Analyst)
                - "How can I secure my network?" (Security Expert)
                - "Create a workout plan" (Wellness Coach)
                """)
            
            # Display chat history
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                agent = message.get("agent", "General Assistant")
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    agent_icon = JARVIS_AGENTS.get(agent, {}).get("icon", "🤖")
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>{agent_icon} JARVIS ({agent}):</strong> {content}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Voice input option
            if VOICE_AVAILABLE and not CLOUD_ENVIRONMENT:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("🎤 Voice Input", use_container_width=True):
                        with st.spinner("🎤 Listening..."):
                            voice_text = listen_for_speech()
                            if voice_text:
                                st.session_state.messages.append({"role": "user", "content": voice_text})
                                st.rerun()
            
            # User text input
            user_input = st.chat_input("Ask JARVIS anything... (or use voice input)")
            
            if user_input:
                # Security validation
                is_valid, validation_msg = validate_input(user_input)
                if not is_valid:
                    st.error(f"🚨 Security Alert: {validation_msg}")
                    log_security_event("invalid_input", {"input": user_input[:100], "reason": validation_msg})
                    return
                
                # Rate limiting check
                rate_ok, rate_msg = check_rate_limiting()
                if not rate_ok:
                    st.warning(f"⚠️ {rate_msg}")
                    log_security_event("rate_limit_exceeded", {"message": rate_msg})
                    return
                
                # Log successful interaction
                log_security_event("user_interaction", {"agent": st.session_state.current_agent})
                
                # Determine which agent to use
                if not st.checkbox("🤖 Auto-select best agent", value=True, key="auto_select_main"):
                    selected_agent = st.session_state.current_agent
                else:
                    selected_agent = auto_select_agent(user_input)
                    st.session_state.current_agent = selected_agent
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {user_input}
                </div>
                """, unsafe_allow_html=True)
                
                # Get AI response
                agent_icon = JARVIS_AGENTS[selected_agent]["icon"]
                with st.spinner(f"🤔 {agent_icon} JARVIS ({selected_agent}) is thinking..."):
                    response = get_gemini_response(user_input, selected_agent)
                    
                    # Sanitize response for security
                    response = sanitize_output(response)
                    
                    # Add AI response with agent info
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "agent": selected_agent
                    })
                    
                    # Display AI response
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>{agent_icon} JARVIS ({selected_agent}):</strong> {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Voice response if enabled
                    if VOICE_AVAILABLE and st.session_state.get('voice_enabled', False) and not CLOUD_ENVIRONMENT:
                        engine = init_speech_engine()
                        if engine:
                            speak_text(response, engine)
        
        with tab2:
            create_daily_agenda()
        
        with tab3:
            create_health_dashboard()
        
        with tab4:
            create_news_feed()
        
        with tab5:
            st.header("🤖 Available Agents")
            st.write("Choose a specialized agent for your specific needs:")
            
            # Agent grid
            cols = st.columns(3)
            for i, (agent_name, agent_data) in enumerate(JARVIS_AGENTS.items()):
                with cols[i % 3]:
                    is_current = agent_name == st.session_state.current_agent
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        f"{agent_data['icon']} {agent_name}",
                        key=f"agent_{agent_name}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_agent = agent_name
                        st.rerun()
                    
                    if is_current:
                        st.success("✅ Active")
                    
                    st.caption(agent_data['description'])
        
        # Continue with chat interface in Tab 1
        # (Moving the existing chat logic to be inside Tab 1)
        
    # Main chat interface (this will be inside Tab 1)
    if st.session_state.gemini_initialized:
        # This section will be moved inside Tab 1, but keeping structure for now
        pass
    else:
        # JARVIS Activation (existing code)
        st.header("🚀 Activate JARVIS")
        
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Activate JARVIS</h1>
            <p>Powered by Gemini Pro with Voice Interface & Specialized Agents</p>
        </div>
        """, unsafe_allow_html=True)
        
        *JARVIS will automatically switch agents based on your requests, or you can manually select one in the sidebar.*
        """)
        
        # Welcome message if no chat history
        if not st.session_state.messages:
            st.success("""
            🎉 **JARVIS V3 is online and ready!**
            
            **Available Capabilities:**
            - 💬 **Chat**: Ask me anything using Gemini Pro intelligence
            - 🎤 **Voice**: Use voice input and get spoken responses  
            - 🎯 **Specialized Agents**: I automatically become the right expert for your task
            - � **Agent Command**: Tell me to "switch to [agent name]" to change modes
            
            **Available Agents:** """ + " • ".join([f"{info['icon']} {name}" for name, info in JARVIS_AGENTS.items()]) + """
            
            Try saying: "Help me plan a project" or "Write a blog post" or "Analyze my budget"
            """)
    else:
        st.header("🔧 Activate JARVIS")
        st.info("""
        **To activate JARVIS V3:**
        
        1. 🔑 Get a free Gemini API key: https://makersuite.google.com/app/apikey
        2. 📝 Paste it in the sidebar under "Gemini Pro Configuration"
        3. 🚀 Start chatting with voice and specialized agents!
        
        **What makes JARVIS V3 special:**
        - 🆓 **Completely free** with Gemini Pro
        - 🎤 **Voice interface** for natural interaction
        - 🎯 **9 specialized agents** for different tasks
        - 🧠 **Smart agent switching** based on your needs
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            agent_used = message.get("agent", "JARVIS")
            agent_icon = JARVIS_AGENTS.get(agent_used, {}).get("icon", "🤖")
            st.markdown(f"""
            <div class="chat-message jarvis-message">
                <strong>{agent_icon} JARVIS ({agent_used}):</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Handle voice input
    voice_input = st.session_state.get('voice_input', '')
    if voice_input:
        user_input = voice_input
        st.session_state.voice_input = ''  # Clear after use
    else:
        # User text input
        if st.session_state.gemini_initialized:
            user_input = st.chat_input("Ask JARVIS anything... (or use voice input)")
        else:
            user_input = st.chat_input("Activate JARVIS first with your Gemini API key...")
    
    if user_input and st.session_state.gemini_initialized:
        # Security validation
        is_valid, validation_msg = validate_input(user_input)
        if not is_valid:
            st.error(f"🚨 Security Alert: {validation_msg}")
            log_security_event("invalid_input", {"input": user_input[:100], "reason": validation_msg})
            return
        
        # Rate limiting check
        rate_ok, rate_msg = check_rate_limiting()
        if not rate_ok:
            st.warning(f"⚠️ {rate_msg}")
            log_security_event("rate_limit_exceeded", {"message": rate_msg})
            return
        
        # Log successful interaction
        log_security_event("user_interaction", {"agent": st.session_state.current_agent})
        
        # Determine which agent to use
        if not st.checkbox("🤖 Auto-select best agent", value=True, key="auto_select_main"):
            selected_agent = st.session_state.current_agent
        else:
            selected_agent = auto_select_agent(user_input)
            st.session_state.current_agent = selected_agent
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # Get AI response
        agent_icon = JARVIS_AGENTS[selected_agent]["icon"]
        with st.spinner(f"🤔 {agent_icon} JARVIS ({selected_agent}) is thinking..."):
            response = get_gemini_response(user_input, selected_agent)
            
            # Sanitize response for security
            response = sanitize_output(response)
            
            # Add AI response with agent info
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "agent": selected_agent
            })
            
            # Display AI response
            st.markdown(f"""
            <div class="chat-message jarvis-message">
                <strong>{agent_icon} JARVIS ({selected_agent}):</strong> {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Voice output
            if voice_engine and st.session_state.get('voice_enabled', False):
                with st.spinner("🔊 JARVIS is speaking..."):
                    speak_text(response, voice_engine)
            
        st.rerun()

if __name__ == "__main__":
    main()