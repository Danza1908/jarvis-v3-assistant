import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from datetime import datetime
import json

# Import the new JARVIS Controller with specialized agents
from jarvis_controller import JarvisController, create_jarvis_controller

# Machine Learning & Data Science imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import joblib
from io import StringIO

# Free AI Alternative imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="JARVIS V3 - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .jarvis-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def initialize_free_jarvis(provider="ollama", model_name="llama2"):
    """Initialize JARVIS with free AI providers"""
    
    try:
        if provider == "ollama" and OLLAMA_AVAILABLE:
            return initialize_ollama_jarvis(model_name)
        elif provider == "huggingface" and TRANSFORMERS_AVAILABLE:
            return initialize_huggingface_jarvis(model_name)
        elif provider == "gemini" and GEMINI_AVAILABLE:
            return initialize_gemini_jarvis()
        elif provider == "groq" and GROQ_AVAILABLE:
            return initialize_groq_jarvis()
        else:
            st.error(f"❌ {provider} is not available or not installed")
            return None, None, None, None
            
    except Exception as e:
        st.error(f"Error initializing {provider}: {str(e)}")
        return None, None, None, None

def initialize_ollama_jarvis(model_name="llama2"):
    """Initialize JARVIS with Ollama (local LLM)"""
    try:
        # Check if Ollama is running
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        if not available_models:
            st.warning("🦙 **Ollama Setup Required**")
            st.info("""
            **To use Ollama (completely free local AI):**
            
            1. **Install Ollama**: https://ollama.ai
            2. **Pull a model**: `ollama pull llama2`
            3. **Available models**: llama2, codellama, mistral, phi, etc.
            
            **Benefits:**
            - ✅ Completely free
            - ✅ Runs offline
            - ✅ Privacy-focused
            - ✅ No API limits
            """)
            return None, None, None, None
        
        # Create a custom LLM wrapper for Ollama
        class OllamaLLM:
            def __init__(self, model_name):
                self.model_name = model_name
            
            def invoke(self, prompt):
                response = ollama.generate(model=self.model_name, prompt=prompt)
                return response['response']
        
        llm = OllamaLLM(model_name)
        
        # Create AI Agents
        research_agent = Agent(
            role="Research Specialist",
            goal="Conduct thorough research and provide accurate information",
            backstory="You are an expert researcher with vast knowledge.",
            llm=llm,
            verbose=True
        )
        
        analysis_agent = Agent(
            role="Data Analyst", 
            goal="Analyze information and provide insights",
            backstory="You are a skilled analyst who identifies patterns and trends.",
            llm=llm,
            verbose=True
        )
        
        assistant_agent = Agent(
            role="Personal Assistant",
            goal="Provide helpful assistance with various tasks",
            backstory="You are JARVIS, an advanced AI assistant.",
            llm=llm,
            verbose=True
        )
        
        ml_agent = Agent(
            role="Machine Learning Specialist",
            goal="Perform ML tasks and data analysis",
            backstory="You are an expert data scientist and ML engineer.",
            llm=llm,
            verbose=True
        )
        
        st.success(f"✅ Ollama initialized with {model_name}!")
        return research_agent, analysis_agent, assistant_agent, ml_agent
        
    except Exception as e:
        st.error(f"Ollama error: {str(e)}")
        return None, None, None, None

def initialize_gemini_jarvis():
    """Initialize JARVIS with Google Gemini (free tier)"""
    try:
        # Check for Gemini API key
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
        
        if not api_key:
            st.warning("🔑 **Gemini API Key Required**")
            st.info("""
            **To use Google Gemini (free tier):**
            
            1. **Get API Key**: https://makersuite.google.com/app/apikey
            2. **Free Tier**: 60 requests per minute
            3. **Add key to sidebar** or environment variable
            
            **Benefits:**
            - ✅ Free tier available
            - ✅ High-quality responses
            - ✅ Fast inference
            """)
            return None, None, None, None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create custom LLM wrapper
        class GeminiLLM:
            def __init__(self):
                self.model = genai.GenerativeModel('gemini-pro')
            
            def invoke(self, prompt):
                response = self.model.generate_content(prompt)
                return response.text
        
        llm = GeminiLLM()
        
        # Create agents (similar structure as above)
        research_agent = Agent(
            role="Research Specialist",
            goal="Conduct thorough research and provide accurate information",
            backstory="You are an expert researcher.",
            llm=llm,
            verbose=True
        )
        
        # ... (similar agent creation as Ollama)
        
        st.success("✅ Google Gemini initialized!")
        return research_agent, None, None, None  # Simplified for now
        
    except Exception as e:
        st.error(f"Gemini error: {str(e)}")
        return None, None, None, None

def initialize_jarvis_with_provider(provider: str, api_key: str):
    """Initialize JARVIS with specified AI provider"""
    try:
        # Initialize the specialized JARVIS Controller with chosen provider
        st.info(f"🔧 Initializing JARVIS V3 with {provider.upper()} and specialized agents...")
        controller = create_jarvis_controller(api_key=api_key, provider=provider)
        
        if controller:
            st.success(f"✅ JARVIS V3 Controller initialized with {provider.upper()} and 9 specialized agents!")
            # Store individual agents for backward compatibility
            agents = {
                'project_manager': controller.get_agent('project_manager'),
                'content_creator': controller.get_agent('content_creator'),
                'financial_analyst': controller.get_agent('financial_analyst'),
                'client_manager': controller.get_agent('client_manager'),
                'web_designer': controller.get_agent('web_designer'),
                'graphic_designer': controller.get_agent('graphic_designer'),
                'wellness_coach': controller.get_agent('wellness_coach'),
                'growth_strategist': controller.get_agent('growth_strategist'),
                'ai_educator': controller.get_agent('ai_educator')
            }
            return controller, agents
        else:
            return None
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if "api" in error_msg or "key" in error_msg:
            st.error(f"🔑 **{provider.upper()} API Key Issue**")
            st.info(f"""
            **There's an issue with your {provider.upper()} API key:**
            
            1. **Verify Key**: Make sure your API key is correct
            2. **Check Permissions**: Ensure the key has proper permissions
            3. **Regenerate**: Try creating a new API key
            
            **Get your API key**: {
                'https://makersuite.google.com/app/apikey' if provider == 'gemini' 
                else 'https://platform.openai.com/api-keys'
            }
            """)
        else:
            st.error(f"❌ **Error initializing JARVIS with {provider.upper()}**: {str(e)}")
            st.info("**You can still use the offline ML features while we resolve this.**")
        
        return None

def initialize_basic_gemini_mode(api_key: str):
    """Initialize basic Gemini chat mode without specialized agents"""
    try:
        if GEMINI_AVAILABLE:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Test the connection
            model = genai.GenerativeModel('gemini-pro')
            test_response = model.generate_content("Hello")
            
            if test_response:
                st.session_state.gemini_model = model
                return True
        return False
    except Exception as e:
        st.error(f"Basic Gemini initialization failed: {str(e)}")
        return False

def initialize_jarvis():
    """Initialize JARVIS AI system with specialized agent controller"""
    
    # Check for OpenAI API key in multiple sources
    api_key = None
    
    # 1. Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 2. Check session state (from sidebar input)
    if not api_key and hasattr(st.session_state, 'api_key') and st.session_state.api_key:
        api_key = st.session_state.api_key
    
    # 3. Try to get from Streamlit secrets if available
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""
    
    if not api_key:
        st.error("🔑 **OpenAI API Key Required**")
        st.info("""
        **To use JARVIS, please provide your OpenAI API key:**
        
        **Option 1:** Enter it in the sidebar ⬅️
        
        **Option 2:** Add it to `.streamlit/secrets.toml`:
        ```
        OPENAI_API_KEY = "your-api-key-here"
        ```
        
        **Option 3:** Set environment variable:
        ```
        export OPENAI_API_KEY="your-api-key-here"
        ```
        
        🔗 **Get your API key:** https://platform.openai.com/api-keys
        """)
        return None
    
    try:
        # Initialize the specialized JARVIS Controller
        st.info("🔧 Initializing JARVIS V3 with specialized agents...")
        controller = create_jarvis_controller(api_key=api_key)
        
        if controller:
            st.success("✅ JARVIS V3 Controller initialized with 9 specialized agents!")
            # Store individual agents for backward compatibility
            agents = {
                'project_manager': controller.get_agent('project_manager'),
                'content_creator': controller.get_agent('content_creator'),
                'financial_analyst': controller.get_agent('financial_analyst'),
                'client_manager': controller.get_agent('client_manager'),
                'web_designer': controller.get_agent('web_designer'),
                'graphic_designer': controller.get_agent('graphic_designer'),
                'wellness_coach': controller.get_agent('wellness_coach'),
                'growth_strategist': controller.get_agent('growth_strategist'),
                'ai_educator': controller.get_agent('ai_educator')
            }
            return controller, agents
        else:
            return None
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "ratelimiterror" in error_msg or "quota" in error_msg:
            st.error("🚨 **OpenAI API Quota Exceeded**")
            st.info("""
            **Your OpenAI API quota has been exceeded. Here's how to fix this:**
            
            1. **Check Your Usage**: Visit [OpenAI Platform Usage](https://platform.openai.com/usage)
            2. **Add Billing**: Go to [Billing Settings](https://platform.openai.com/account/billing) 
            3. **Upgrade Plan**: Consider upgrading to a paid plan for higher limits
            4. **Wait for Reset**: Free tier quotas reset monthly
            
            **Alternative Options:**
            - Use the offline ML features (data analysis, visualizations)
            - Upload CSV data and use local analysis tools
            - Try again later when your quota resets
            
            **For now, you can still use:**
            ✅ Data upload and analysis
            ✅ Statistical reporting  
            ✅ Data visualizations
            ✅ ML model suggestions (offline)
            """)
        elif "authentication" in error_msg or "api" in error_msg:
            st.error("🔑 **API Key Issue**")
            st.info("""
            **There's an issue with your API key:**
            
            1. **Verify Key**: Make sure your API key is correct and starts with 'sk-'
            2. **Check Permissions**: Ensure the key has proper permissions
            3. **Regenerate**: Try creating a new API key
            
            **Get your API key**: https://platform.openai.com/api-keys
            """)
        else:
            st.error(f"❌ **Error initializing JARVIS**: {str(e)}")
            st.info("**You can still use the offline ML features while we resolve this.**")
        
        return None

def create_task_and_crew(user_input, controller, agents=None):
    """Create appropriate task and crew based on user input using specialized agents"""
    
    # Check if user has selected a specific agent
    selected_agent = st.session_state.get('selected_agent', 'auto')
    
    if selected_agent != 'auto':
        # Use the specifically selected agent
        return controller.execute_task(user_input, selected_agent)
    
    # Auto-select based on user input keywords
    user_input_lower = user_input.lower()
    
    # Project Management keywords
    if any(keyword in user_input_lower for keyword in ["project", "manage", "plan", "timeline", "schedule", "task", "deadline", "milestone", "organize", "coordinate"]):
        return controller.execute_task(user_input, "project_manager")
    
    # Content Creation keywords
    elif any(keyword in user_input_lower for keyword in ["write", "content", "article", "blog", "story", "script", "copy", "create text", "writing", "draft"]):
        return controller.execute_task(user_input, "content_creator")
    
    # Financial Analysis keywords
    elif any(keyword in user_input_lower for keyword in ["finance", "financial", "budget", "investment", "cost", "revenue", "profit", "expense", "roi", "money", "pricing"]):
        return controller.execute_task(user_input, "financial_analyst")
    
    # Client Management keywords
    elif any(keyword in user_input_lower for keyword in ["client", "customer", "relationship", "communication", "service", "support", "crm", "contact"]):
        return controller.execute_task(user_input, "client_manager")
    
    # Web Design keywords
    elif any(keyword in user_input_lower for keyword in ["web", "website", "ui", "ux", "design", "interface", "frontend", "responsive", "html", "css", "javascript"]):
        return controller.execute_task(user_input, "web_designer")
    
    # Graphic Design keywords
    elif any(keyword in user_input_lower for keyword in ["graphic", "logo", "visual", "image", "graphics", "design", "brand", "poster", "flyer", "banner"]):
        return controller.execute_task(user_input, "graphic_designer")
    
    # Wellness keywords
    elif any(keyword in user_input_lower for keyword in ["wellness", "health", "fitness", "exercise", "nutrition", "meditation", "stress", "wellbeing", "mental health"]):
        return controller.execute_task(user_input, "wellness_coach")
    
    # Growth Strategy keywords
    elif any(keyword in user_input_lower for keyword in ["growth", "strategy", "marketing", "scale", "expand", "business development", "market", "competition"]):
        return controller.execute_task(user_input, "growth_strategist")
    
    # AI/Technology Education keywords
    elif any(keyword in user_input_lower for keyword in ["ai", "artificial intelligence", "machine learning", "ml", "technology", "learn", "education", "tutorial", "explain"]):
        return controller.execute_task(user_input, "ai_educator")
    
    # Machine Learning/Data Science (legacy support)
    elif any(keyword in user_input_lower for keyword in ["model", "train", "predict", "classification", "regression", "cluster", "data science", "statistics", "visualization", "plot", "chart", "graph"]):
        return controller.execute_task(user_input, "ai_educator")  # Route to AI educator for technical explanations
    
    # Research tasks (route to most appropriate agent or AI educator for general knowledge)
    elif any(keyword in user_input_lower for keyword in ["research", "find", "search", "information about", "tell me about"]):
        return controller.execute_task(user_input, "ai_educator")
    
    # Analysis tasks (route to appropriate domain or financial analyst for general analysis)
    elif any(keyword in user_input_lower for keyword in ["analyze", "compare", "evaluate", "pros and cons", "analysis"]):
        return controller.execute_task(user_input, "financial_analyst")
    
    else:
        # Default to AI educator for general assistance
        return controller.execute_task(user_input, "ai_educator")

def generate_data_report(df):
    """Generate comprehensive data analysis report"""
    try:
        report = []
        report.append("# 📊 Data Analysis Report\n")
        
        # Basic info
        report.append(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n")
        
        # Data types
        report.append("## Data Types")
        for col, dtype in df.dtypes.items():
            report.append(f"- **{col}**: {dtype}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            report.append("\n## Missing Values")
            for col, miss in missing.items():
                if miss > 0:
                    report.append(f"- **{col}**: {miss} ({miss/len(df)*100:.1f}%)")
        else:
            report.append("\n## ✅ No missing values found")
        
        # Numerical columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report.append("\n## 📈 Numerical Columns Summary")
            stats_df = df[numeric_cols].describe()
            report.append(stats_df.to_string())
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            report.append("\n## 📋 Categorical Columns")
            for col in cat_cols:
                unique_count = df[col].nunique()
                report.append(f"- **{col}**: {unique_count} unique values")
                if unique_count <= 10:
                    report.append(f"  Values: {list(df[col].unique())}")
        
        return "\n".join(report)
    except Exception as e:
        return f"Error generating report: {str(e)}"

def suggest_ml_models(df):
    """Suggest appropriate ML models based on data characteristics"""
    try:
        suggestions = []
        suggestions.append("# 🎯 ML Model Suggestions\n")
        
        # Analyze data characteristics
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        suggestions.append(f"**Data characteristics:**")
        suggestions.append(f"- {n_rows} samples, {n_cols} features")
        suggestions.append(f"- {len(numeric_cols)} numerical, {len(cat_cols)} categorical features\n")
        
        # Model suggestions based on data size
        if n_rows < 1000:
            suggestions.append("## Recommended for Small Dataset:")
            suggestions.append("- **Linear/Logistic Regression** - Fast, interpretable")
            suggestions.append("- **Decision Trees** - Good for non-linear patterns")
            suggestions.append("- **K-Nearest Neighbors** - Simple, effective for small data")
        elif n_rows < 10000:
            suggestions.append("## Recommended for Medium Dataset:")
            suggestions.append("- **Random Forest** - Robust, handles mixed data types")
            suggestions.append("- **Gradient Boosting** - High performance")
            suggestions.append("- **Support Vector Machines** - Good for complex patterns")
        else:
            suggestions.append("## Recommended for Large Dataset:")
            suggestions.append("- **Random Forest** - Scalable, parallel processing")
            suggestions.append("- **Gradient Boosting (XGBoost/LightGBM)** - State-of-the-art performance")
            suggestions.append("- **Neural Networks** - Deep learning for complex patterns")
        
        # Task-specific suggestions
        suggestions.append("\n## By Problem Type:")
        suggestions.append("**Classification (predicting categories):**")
        suggestions.append("- Random Forest Classifier, Logistic Regression, SVM")
        suggestions.append("\n**Regression (predicting continuous values):**")
        suggestions.append("- Random Forest Regressor, Linear Regression, Gradient Boosting")
        suggestions.append("\n**Clustering (finding groups):**")
        suggestions.append("- K-Means, DBSCAN, Hierarchical Clustering")
        
        return "\n".join(suggestions)
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

def create_visualizations(df):
    """Create data visualizations"""
    try:
        viz_report = []
        viz_report.append("# 📊 Data Visualizations Created\n")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns
        
        # Correlation heatmap for numerical data
        if len(numeric_cols) > 1:
            fig_corr = px.imshow(
                df[numeric_cols].corr(),
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            viz_report.append("✅ **Correlation Heatmap** - Shows relationships between numerical features")
        
        # Distribution plots
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                fig_hist = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            viz_report.append("✅ **Distribution Plots** - Shows data distribution patterns")
        
        # Categorical data
        if len(cat_cols) > 0:
            for col in cat_cols[:2]:  # Limit to first 2 columns
                if df[col].nunique() <= 20:  # Only if not too many categories
                    fig_bar = px.bar(df[col].value_counts().reset_index(), 
                                   x='index', y=col, 
                                   title=f"Count of {col}")
                    st.plotly_chart(fig_bar, use_container_width=True)
            viz_report.append("✅ **Category Counts** - Shows frequency of categorical values")
        
        # Scatter plot if we have at least 2 numerical columns
        if len(numeric_cols) >= 2:
            fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                   title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig_scatter, use_container_width=True)
            viz_report.append("✅ **Scatter Plot** - Shows relationship between two numerical features")
        
        viz_report.append("\n**Visualizations help you understand:**")
        viz_report.append("- Data distribution and patterns")
        viz_report.append("- Relationships between features")
        viz_report.append("- Outliers and anomalies")
        viz_report.append("- Feature importance for modeling")
        
        return "\n".join(viz_report)
    except Exception as e:
        return f"Error creating visualizations: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 JARVIS V3 - Specialized AI Assistant</h1>
        <p>Your intelligent AI companion with 9 specialized domain experts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show specialized agents info
    if st.session_state.get('initialized', False):
        if st.session_state.get('basic_gemini_mode', False):
            st.info("""
            🤖 **Basic Gemini Chat Mode Active**
            
            You're chatting directly with Google Gemini! While specialized agents aren't available in this mode, 
            you still get powerful AI assistance for all your questions.
            
            *For full specialized agent functionality, we're working on CrewAI + Gemini integration.*
            """)
        else:
            st.info("""
            🎯 **9 Specialized AI Agents Available:**
            
            **Business & Strategy:** 📊 Project Manager • 💰 Financial Analyst • 📈 Growth Strategist • 🤝 Client Manager
            
            **Creative & Design:** ✍️ Content Creator • 🎨 Graphic Designer • 🌐 Web Designer
            
            **Personal & Education:** 🧘 Wellness Coach • 🤖 AI Educator
            
            *Ask about anything and I'll route your request to the most qualified specialist!*
            """)
    else:
        st.info("""
        🔧 **Setup Required:** Configure your OpenAI API key in the sidebar to access all 9 specialized AI agents, or use the offline ML tools below.
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # AI Provider Selection
        st.markdown("**🤖 AI Provider (Free Options Available!)**")
        
        provider_options = []
        if OLLAMA_AVAILABLE:
            provider_options.append("🦙 Ollama (Local, Free)")
        if GEMINI_AVAILABLE:
            provider_options.append("💎 Google Gemini (Free Tier)")
        if GROQ_AVAILABLE:
            provider_options.append("🌐 Groq (Free)")
        provider_options.append("🏢 OpenAI (Paid)")
        
        selected_provider = st.selectbox(
            "Choose AI Provider",
            provider_options,
            help="Free options available! No tokens required for Ollama."
        )
        
        # Provider-specific configuration
        if "Ollama" in selected_provider:
            st.info("🆓 **Ollama is completely FREE and runs locally!**")
            if OLLAMA_AVAILABLE:
                try:
                    models = ollama.list()
                    available_models = [model['name'] for model in models['models']]
                    if available_models:
                        selected_model = st.selectbox("Select Ollama Model", available_models)
                        st.session_state.ai_provider = "ollama"
                        st.session_state.ai_model = selected_model
                    else:
                        st.warning("No Ollama models found. Run: `ollama pull llama2`")
                except:
                    st.warning("Ollama not running. Start Ollama service first.")
            else:
                st.warning("Ollama not installed. Visit: https://ollama.ai")
                
        elif "Gemini" in selected_provider:
            st.info("🆓 **Google Gemini has a generous free tier!**")
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password", 
                help="Get free key: https://makersuite.google.com/app/apikey",
                placeholder="AIza..."
            )
            if gemini_key:
                if gemini_key.startswith('AIza') and len(gemini_key) > 20:
                    os.environ["GEMINI_API_KEY"] = gemini_key
                    st.session_state.gemini_key = gemini_key
                    st.session_state.ai_provider = "gemini"
                    st.success("✅ Gemini key added!")
                else:
                    st.error("❌ Invalid Gemini API key format. Should start with 'AIza'")
                
        elif "Groq" in selected_provider:
            st.info("🆓 **Groq offers free high-speed inference!**")
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Get free key: https://console.groq.com",
                placeholder="gsk_..."
            )
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
                st.session_state.ai_provider = "groq"
                st.success("✅ Groq key added!")
                
        else:  # OpenAI
            st.warning("💰 **OpenAI requires paid credits**")
            # API Key input
            st.markdown("**🔑 OpenAI Configuration**")
            api_key_input = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Get your API key from: https://platform.openai.com/api-keys",
                placeholder="sk-..."
            )
            
            if api_key_input:
                if api_key_input.startswith('sk-') and len(api_key_input) > 20:
                    os.environ["OPENAI_API_KEY"] = api_key_input
                    st.session_state.api_key = api_key_input
                    st.session_state.ai_provider = "openai"
                    st.success("✅ API Key Added!")
                else:
                    st.error("❌ Invalid API key format. Should start with 'sk-'")
            
            # Model selection
            model_choice = st.selectbox(
                "Select Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=0
            )
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Specialized Agent Selector
        if st.session_state.get('initialized', False):
            st.markdown("---")
            st.markdown("**🎯 Specialized Agents**")
            
            agent_options = {
                "Auto-Select (Recommended)": "auto",
                "📊 Project Manager": "project_manager",
                "✍️ Content Creator": "content_creator", 
                "💰 Financial Analyst": "financial_analyst",
                "🤝 Client Manager": "client_manager",
                "🌐 Web Designer": "web_designer",
                "🎨 Graphic Designer": "graphic_designer",
                "🧘 Wellness Coach": "wellness_coach",
                "📈 Growth Strategist": "growth_strategist",
                "🤖 AI Educator": "ai_educator"
            }
            
            selected_agent = st.selectbox(
                "Choose Agent",
                list(agent_options.keys()),
                help="Auto-select routes your request to the most appropriate specialist"
            )
            
            st.session_state.selected_agent = agent_options[selected_agent]
            
            if st.session_state.selected_agent != "auto":
                st.info(f"🎯 **Direct routing to:** {selected_agent}")
        
        # Machine Learning Features
        st.markdown("---")
        st.markdown("**🧠 ML & Data Science**")
        
        # Data upload
        uploaded_file = st.file_uploader(
            "Upload CSV Data",
            type=['csv'],
            help="Upload a CSV file for ML analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Quick data preview
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head())
                    st.write("**Data Info:**")
                    buffer = StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        # ML Quick Actions
        if hasattr(st.session_state, 'uploaded_data'):
            st.markdown("**Quick ML Actions:**")
            if st.button("📈 Generate Data Report"):
                st.session_state.ml_action = "data_report"
            if st.button("🎯 Suggest ML Models"):
                st.session_state.ml_action = "model_suggestions"
            if st.button("📊 Create Visualizations"):
                st.session_state.ml_action = "visualizations"
        
        # System info
        st.markdown("---")
        st.markdown("**System Status**")
        
        # AI Provider Status
        current_provider = st.session_state.get('ai_provider', 'none')
        if current_provider == 'ollama':
            st.success("✅ Ollama (FREE Local AI)")
        elif current_provider == 'gemini':
            st.success("✅ Google Gemini (FREE Tier)")
        elif current_provider == 'groq':
            st.success("✅ Groq (FREE)")
        elif current_provider == 'openai' and os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI (Paid)")
        else:
            st.error("❌ No AI Provider Configured")
        
        # ML Libraries status
        try:
            import pandas as pd
            import numpy as np
            import sklearn
            st.success("✅ ML Libraries Loaded")
        except ImportError:
            st.warning("⚠️ Some ML libraries not available")
            
        # Free AI Libraries status
        status_items = []
        if OLLAMA_AVAILABLE:
            status_items.append("🦙 Ollama")
        if GEMINI_AVAILABLE:
            status_items.append("💎 Gemini")
        if GROQ_AVAILABLE:
            status_items.append("🌐 Groq")
            
        if status_items:
            st.info(f"🆓 Available: {', '.join(status_items)}")
    
    # Initialize JARVIS if not already done
    if not st.session_state.initialized:
        with st.spinner("🔧 Initializing JARVIS V3 with specialized agents..."):
            # Determine which provider to use
            provider = st.session_state.get('ai_provider', 'openai')
            
            if provider == "gemini":
                # Use Gemini with specialized agents
                gemini_key = os.getenv("GEMINI_API_KEY") or st.session_state.get('gemini_key')
                if gemini_key:
                    try:
                        result = initialize_jarvis_with_provider("gemini", gemini_key)
                        if result:
                            controller, agents = result
                            st.session_state.controller = controller
                            st.session_state.agents = agents
                            st.session_state.initialized = True
                            st.success("✅ JARVIS V3 initialized with Gemini and 9 specialized agents!")
                        else:
                            st.session_state.initialized = False
                            st.session_state.offline_mode = True
                            st.warning("⚠️ Failed to initialize with Gemini. Using fallback mode.")
                    except Exception as e:
                        st.error(f"Gemini initialization error: {str(e)}")
                        # Fallback: Try to initialize basic agents without specialized controller
                        st.info("🔄 Falling back to basic Gemini chat mode...")
                        if initialize_basic_gemini_mode(gemini_key):
                            st.session_state.initialized = True
                            st.session_state.basic_gemini_mode = True
                            st.success("✅ Basic Gemini chat mode enabled!")
                        else:
                            st.session_state.initialized = False
                            st.session_state.offline_mode = True
                            st.warning("⚠️ Gemini initialization failed completely.")
                else:
                    st.session_state.initialized = False
                    st.session_state.offline_mode = True
                    st.warning("⚠️ Gemini API key required. Add it in the sidebar.")
                    
            elif provider == "openai":
                # Use OpenAI with specialized agents
                result = initialize_jarvis()
                if result:
                    controller, agents = result
                    st.session_state.controller = controller
                    st.session_state.agents = agents
                    st.session_state.initialized = True
                    st.success("✅ JARVIS V3 initialized with OpenAI and 9 specialized agents!")
                else:
                    st.session_state.initialized = False
                    st.session_state.offline_mode = True
                    st.warning("⚠️ JARVIS AI chat is unavailable, but you can still use offline ML features!")
            else:
                # Default fallback
                st.session_state.initialized = False
                st.session_state.offline_mode = True
                st.info("🔄 Please select an AI provider in the sidebar to enable specialized agents.")
    
    # Chat interface
    if st.session_state.get('initialized', False):
        st.header("💬 Chat with JARVIS")
    else:
        st.header("🔧 Offline Mode - ML Tools Available")
        st.info("""
        **While AI chat is unavailable, you can still:**
        - 📊 Upload CSV data for analysis
        - 📈 Generate data reports and visualizations  
        - 🎯 Get ML model recommendations
        - 📋 Perform statistical analysis
        
        Use the sidebar tools or fix your API key to enable full chat functionality.
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
            st.markdown(f"""
            <div class="chat-message jarvis-message">
                <strong>JARVIS:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    if st.session_state.get('initialized', False):
        user_input = st.chat_input("Ask JARVIS anything...")
    else:
        user_input = st.chat_input("AI chat unavailable - use sidebar ML tools or fix API key...")
    
    if user_input:
        if st.session_state.get('initialized', False):
            # Normal AI chat mode
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {user_input}
            </div>
            """, unsafe_allow_html=True)
            
            # Process with JARVIS
            with st.spinner("🤔 JARVIS is thinking..."):
                try:
                    # Check if we're in basic Gemini mode
                    if st.session_state.get('basic_gemini_mode', False) and hasattr(st.session_state, 'gemini_model'):
                        # Use basic Gemini mode
                        gemini_response = st.session_state.gemini_model.generate_content(user_input)
                        response = gemini_response.text
                        
                    # Use the specialized controller
                    elif hasattr(st.session_state, 'controller') and st.session_state.controller:
                        response = create_task_and_crew(user_input, st.session_state.controller, st.session_state.agents)
                    else:
                        response = "I apologize, but the JARVIS controller is not properly initialized. Please check your API key and restart the application."
                    
                    # Add JARVIS response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display JARVIS response
                    st.markdown(f"""
                    <div class="chat-message jarvis-message">
                        <strong>JARVIS:</strong> {response}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
        else:
            # Offline mode - provide helpful message
            st.info("💡 **AI chat is currently unavailable.** Please use the ML tools in the sidebar or resolve your API key issue to enable chat functionality.")
    
    # Handle ML quick actions
    if hasattr(st.session_state, 'ml_action') and hasattr(st.session_state, 'uploaded_data'):
        action = st.session_state.ml_action
        df = st.session_state.uploaded_data
        
        if action == "data_report":
            with st.spinner("📊 Generating data report..."):
                report = generate_data_report(df)
                st.markdown(report)
                st.session_state.messages.append({"role": "assistant", "content": report})
                
        elif action == "model_suggestions":
            with st.spinner("🎯 Analyzing data for ML suggestions..."):
                suggestions = suggest_ml_models(df)
                st.markdown(suggestions)
                st.session_state.messages.append({"role": "assistant", "content": suggestions})
                
        elif action == "visualizations":
            with st.spinner("📊 Creating visualizations..."):
                viz_report = create_visualizations(df)
                st.markdown(viz_report)
                st.session_state.messages.append({"role": "assistant", "content": viz_report})
        
        # Clear the action
        del st.session_state.ml_action
        st.rerun()
    
    # Welcome message for new users
    if not st.session_state.messages and st.session_state.initialized:
        current_provider = st.session_state.get('ai_provider', 'none')
        provider_msg = ""
        if current_provider == 'ollama':
            provider_msg = "🆓 **Running on FREE Ollama (Local AI)** - No tokens required!"
        elif current_provider == 'gemini':
            provider_msg = "🆓 **Using FREE Google Gemini** - Generous free tier!"
        elif current_provider == 'groq':
            provider_msg = "🆓 **Using FREE Groq** - Lightning fast inference!"
        
        st.success(f"""
        👋 **Welcome to JARVIS V3 - Now with FREE AI Options!** 
        
        {provider_msg}
        
        **🆓 FREE AI PROVIDERS AVAILABLE:**
        - 🦙 **Ollama** - Completely free, runs locally, no limits
        - 💎 **Google Gemini** - Free tier with 60 requests/minute  
        - 🌐 **Groq** - Super fast free inference
        
        **🚀 What I can help you with:**
        - 🔍 **Research** and information gathering
        - 📊 **Data analysis** and statistical insights
        - 🧠 **Machine Learning** - model training, predictions, recommendations
        - 📈 **Data Visualization** - charts, plots, dashboards
        - 🎯 **Predictive Analytics** - classification, regression, clustering
        - 💡 **Problem-solving** and strategic recommendations
        - 📝 **Writing** and content creation
        
        **�️ ML Features (Always Free):**
        - Upload CSV data in the sidebar for instant analysis
        - Ask questions like: "train a model", "create visualizations", "analyze this data"
        - Get ML model recommendations based on your data
        - Generate comprehensive data reports and insights
        
        **💡 No OpenAI credits needed! Choose a free provider in the sidebar.**
        """)

if __name__ == "__main__":
    main()
