"""
JARVIS V3 Controller - Core Orchestrator System
===============================================

This module defines the comprehensive JARVIS V3 system with specialized AI agents
for different business and personal domains. Each agent is designed to handle
specific tasks within their expertise area.

Author: JARVIS V3 Team
Version: 3.0
"""

from crewai import Agent, Crew, Process, LLM
import os
from typing import List, Optional
from datetime import datetime

# Google AI imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import aiplatform
    from google.cloud import bigquery
    GOOGLE_ML_AVAILABLE = True
except ImportError:
    GOOGLE_ML_AVAILABLE = False


class JarvisController:
    """
    Main controller class for JARVIS V3 orchestrator system.
    Manages all specialized agents and coordinates their activities.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", provider: str = "openai"):
        """
        Initialize the JARVIS V3 Controller with all specialized agents.
        
        Args:
            api_key (str, optional): API key for chosen provider. If None, will try to get from environment.
            model (str): Model to use (default: gpt-3.5-turbo for OpenAI, gemini-pro for Gemini)
            provider (str): AI provider to use ('openai', 'gemini')
        """
        self.provider = provider.lower()
        self.model = model
        
        # Initialize LLM based on provider
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ValueError("Google Generative AI not available. Install with: pip install google-generativeai")
            
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Use Gemini model with CrewAI
            if model == "gpt-3.5-turbo":  # Default was OpenAI, switch to Gemini default
                self.model = "gemini-pro"
            
            # Note: CrewAI might not support Gemini directly yet, so we'll use a workaround
            try:
                self.llm = LLM(model=f"gemini/{self.model}", api_key=self.api_key)
            except Exception as e:
                # Fallback: Use OpenAI format but with Gemini backend
                print(f"Direct Gemini integration failed: {e}")
                self.llm = LLM(model="gpt-3.5-turbo", api_key=self.api_key, base_url="https://generativelanguage.googleapis.com")
            
        else:  # Default to OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
            self.llm = LLM(model=self.model, api_key=self.api_key)
        
        # Initialize agent storage
        self.agents = {}
        self.crew = None
        
        # Initialize all specialized agents
        if self.llm:
            self._initialize_agents()
            self._create_crew()
        else:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    def _initialize_agents(self):
        """Initialize all specialized agents with their roles, goals, and backstories."""
        
        # 1. Project Management Agent
        self.agents['project_manager'] = Agent(
            role='Project Manager',
            goal='Create, organize, and manage project plans for business goals.',
            backstory="""You are an experienced project manager with over 10 years in the field. 
            You excel at breaking down complex business objectives into manageable tasks, 
            creating realistic timelines, and coordinating resources efficiently. You understand 
            various project management methodologies including Agile, Scrum, and Waterfall, 
            and can adapt your approach based on project requirements. Your expertise includes 
            risk assessment, stakeholder management, and ensuring projects are delivered on time 
            and within budget.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 2. Content Creation Agent
        self.agents['content_creator'] = Agent(
            role='Senior Content Creator',
            goal='Generate high-quality blog posts, social media content, and marketing copy.',
            backstory="""You are a seasoned content creator and copywriter with expertise in 
            digital marketing and brand storytelling. You have a deep understanding of SEO, 
            content strategy, and audience engagement across multiple platforms. Your writing 
            style is versatile, adapting to different brand voices and target demographics. 
            You stay current with content trends, platform algorithms, and best practices 
            for maximizing reach and engagement. Your content consistently drives results 
            and builds meaningful connections between brands and their audiences.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 3. Financial Agent
        self.agents['financial_analyst'] = Agent(
            role='Financial Analyst',
            goal='Provide accurate financial reports, track income/expenses, and offer strategic advice.',
            backstory="""You are a certified financial analyst with extensive experience in 
            financial planning, analysis, and strategic advisory. You have a strong background 
            in accounting principles, financial modeling, and investment analysis. You excel 
            at interpreting financial data, identifying trends, and providing actionable insights 
            for business growth and personal wealth management. Your expertise includes budgeting, 
            cash flow analysis, risk assessment, and financial forecasting. You stay updated 
            with market trends, regulatory changes, and economic indicators that impact 
            financial decisions.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 4. Client Communication Agent
        self.agents['client_manager'] = Agent(
            role='Client Relationship Manager',
            goal='Draft and refine professional email communications and client updates.',
            backstory="""You are an expert in client relationship management with a talent 
            for professional communication. You understand the nuances of business etiquette, 
            cultural sensitivity, and effective stakeholder engagement. Your communication 
            style is clear, professional, and empathetic, always maintaining positive 
            relationships while addressing complex issues. You have experience managing 
            client expectations, resolving conflicts diplomatically, and ensuring client 
            satisfaction. You're skilled at translating technical information into 
            client-friendly language and maintaining long-term business relationships.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 5. Web Design Agent
        self.agents['web_designer'] = Agent(
            role='Creative Web Designer',
            goal='Develop project plans for modern, user-friendly websites.',
            backstory="""You are a creative web designer with expertise in UX/UI design, 
            front-end development, and digital user experience. You have a strong understanding 
            of modern web technologies, responsive design principles, and accessibility standards. 
            Your design philosophy focuses on creating intuitive, visually appealing websites 
            that deliver exceptional user experiences. You stay current with design trends, 
            web technologies, and best practices for performance optimization. You excel at 
            translating business requirements into compelling digital experiences that drive 
            user engagement and conversion.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 6. Graphic Design Agent
        self.agents['graphic_designer'] = Agent(
            role='Visual Branding Specialist',
            goal='Create project plans for professional logos, visual assets, and branding kits.',
            backstory="""You are a talented visual branding specialist with extensive experience 
            in graphic design, brand identity development, and visual communication. You have 
            a keen eye for color theory, typography, and composition, and understand how visual 
            elements influence brand perception and customer behavior. Your expertise spans 
            logo design, brand guidelines, marketing materials, and digital assets. You excel 
            at creating cohesive visual identities that effectively communicate brand values 
            and resonate with target audiences. You stay updated with design software, 
            printing technologies, and current design trends.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 7. Health Agent
        self.agents['wellness_coach'] = Agent(
            role='Personal Wellness Coach',
            goal='Create and provide personalized plans for fitness, nutrition, and well-being.',
            backstory="""You are a certified personal wellness coach with expertise in fitness, 
            nutrition, and holistic health management. You have formal training in exercise 
            science, nutritional counseling, and behavioral psychology. You understand the 
            interconnection between physical health, mental well-being, and lifestyle factors. 
            Your approach is evidence-based, personalized, and sustainable, focusing on long-term 
            health improvements rather than quick fixes. You excel at creating realistic, 
            achievable wellness plans that fit into busy lifestyles and adapt to individual 
            needs, preferences, and health conditions.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 8. Personal Development Agent
        self.agents['growth_strategist'] = Agent(
            role='Growth Strategist',
            goal='Provide strategies and resources for professional and personal growth.',
            backstory="""You are an experienced personal development coach and growth strategist 
            with a background in psychology, leadership development, and career coaching. You 
            have helped hundreds of individuals achieve their personal and professional goals 
            through strategic planning, skill development, and mindset transformation. Your 
            expertise includes goal setting, habit formation, time management, leadership skills, 
            and emotional intelligence. You understand the psychology of change and motivation, 
            and you excel at creating personalized development plans that align with individual 
            values, strengths, and aspirations.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # 9. AI Education Agent
        self.agents['ai_educator'] = Agent(
            role='AI Systems Educator',
            goal='Provide clear explanations of complex AI and ML concepts and assist with academic research.',
            backstory="""You are an AI systems educator and researcher with deep expertise in 
            artificial intelligence, machine learning, and data science. You hold advanced 
            degrees in computer science and have extensive experience in both academic research 
            and practical AI implementation. You excel at breaking down complex technical 
            concepts into understandable explanations for students and professionals at various 
            levels. Your teaching style is patient, thorough, and engaging, using real-world 
            examples and analogies to make difficult concepts accessible. You stay current 
            with the latest developments in AI research, emerging technologies, and educational 
            methodologies.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def _create_crew(self):
        """Create the crew with all agents using sequential processing."""
        
        # Get all agents as a list
        all_agents = list(self.agents.values())
        
        # Create the crew with sequential process
        self.crew = Crew(
            agents=all_agents,
            tasks=[],  # Tasks will be added dynamically based on requests
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True
        )
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Get a specific agent by name.
        
        Args:
            agent_name (str): Name of the agent to retrieve
            
        Returns:
            Agent: The requested agent or None if not found
        """
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """
        Get a list of all available agent names.
        
        Returns:
            List[str]: List of agent names
        """
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> dict:
        """
        Get detailed information about a specific agent.
        
        Args:
            agent_name (str): Name of the agent
            
        Returns:
            dict: Agent information including role, goal, and backstory
        """
        agent = self.agents.get(agent_name)
        if agent:
            return {
                'name': agent_name,
                'role': agent.role,
                'goal': agent.goal,
                'backstory': agent.backstory
            }
        return {}
    
    def get_all_agents_info(self) -> dict:
        """
        Get information about all agents.
        
        Returns:
            dict: Dictionary with all agent information
        """
        return {name: self.get_agent_info(name) for name in self.agents.keys()}
    
    def __str__(self) -> str:
        """String representation of the JARVIS Controller."""
        return f"JARVIS V3 Controller - {len(self.agents)} specialized agents ready"
    
    def __repr__(self) -> str:
        """Detailed representation of the JARVIS Controller."""
        return f"JarvisController(model='{self.model}', agents={len(self.agents)})"


# Example usage and initialization
def create_jarvis_controller(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> JarvisController:
    """
    Factory function to create a JARVIS Controller instance.
    
    Args:
        api_key (str, optional): OpenAI API key
        model (str): LLM model to use
        
    Returns:
        JarvisController: Initialized controller instance
    """
    try:
        controller = JarvisController(api_key=api_key, model=model)
        print(f"✅ JARVIS V3 Controller initialized successfully!")
        print(f"📊 Available agents: {', '.join(controller.list_agents())}")
        return controller
    except Exception as e:
        print(f"❌ Failed to initialize JARVIS Controller: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Main execution block for testing the JARVIS Controller.
    """
    print("🤖 JARVIS V3 Controller - Specialized AI Agent System")
    print("=" * 60)
    
    try:
        # Initialize the controller
        jarvis = create_jarvis_controller()
        
        # Display agent information
        print("\n📋 Agent Portfolio:")
        for agent_name in jarvis.list_agents():
            info = jarvis.get_agent_info(agent_name)
            print(f"\n🎯 {info['role']}")
            print(f"   Goal: {info['goal']}")
        
        print(f"\n🚀 {jarvis}")
        print("   Ready to execute tasks with specialized agents!")
        
    except Exception as e:
        print(f"⚠️  Error: {str(e)}")
        print("💡 Make sure to set your API key environment variable.")


def create_jarvis_controller(api_key: str = None, provider: str = "openai", model: str = None):
    """
    Create JARVIS Controller with specified AI provider
    
    Args:
        api_key (str): API key for the chosen provider
        provider (str): AI provider ('openai' or 'gemini')
        model (str): Model to use (optional, will use provider defaults)
    
    Returns:
        JarvisController: Initialized controller instance
    """
    try:
        if provider.lower() == "gemini":
            if not model:
                model = "gemini-pro"
            controller = JarvisController(api_key=api_key, provider="gemini", model=model)
        else:  # Default to OpenAI
            if not model:
                model = "gpt-3.5-turbo"
            controller = JarvisController(api_key=api_key, provider="openai", model=model)
        
        return controller
    except Exception as e:
        print(f"Failed to create JARVIS controller: {str(e)}")
        return None