# JARVIS V3 Controller Usage Guide

## 🤖 Overview

The JARVIS V3 Controller (`jarvis_controller.py`) is a comprehensive orchestrator system that manages 9 specialized AI agents, each designed for specific business and personal domains.

## 🎯 Available Agents

### 1. **Project Manager** 
- **Role**: Project Manager
- **Goal**: Create, organize, and manage project plans for business goals
- **Best For**: Task planning, timeline creation, resource management

### 2. **Senior Content Creator**
- **Role**: Senior Content Creator  
- **Goal**: Generate high-quality blog posts, social media content, and marketing copy
- **Best For**: Blog writing, social media, marketing materials

### 3. **Financial Analyst**
- **Role**: Financial Analyst
- **Goal**: Provide accurate financial reports, track income/expenses, and offer strategic advice
- **Best For**: Financial planning, budgeting, investment advice

### 4. **Client Relationship Manager**
- **Role**: Client Relationship Manager
- **Goal**: Draft and refine professional email communications and client updates
- **Best For**: Professional emails, client communication, relationship management

### 5. **Creative Web Designer**
- **Role**: Creative Web Designer
- **Goal**: Develop project plans for modern, user-friendly websites
- **Best For**: Website planning, UX/UI design, web strategy

### 6. **Visual Branding Specialist**
- **Role**: Visual Branding Specialist
- **Goal**: Create project plans for professional logos, visual assets, and branding kits
- **Best For**: Logo design, branding, visual identity

### 7. **Personal Wellness Coach**
- **Role**: Personal Wellness Coach
- **Goal**: Create and provide personalized plans for fitness, nutrition, and well-being
- **Best For**: Fitness plans, nutrition advice, wellness strategies

### 8. **Growth Strategist**
- **Role**: Growth Strategist
- **Goal**: Provide strategies and resources for professional and personal growth
- **Best For**: Career development, skill building, personal growth

### 9. **AI Systems Educator**
- **Role**: AI Systems Educator
- **Goal**: Provide clear explanations of complex AI and ML concepts and assist with academic research
- **Best For**: AI education, research assistance, technical explanations

## 🚀 Usage Examples

### Basic Initialization

```python
from jarvis_controller import create_jarvis_controller

# Initialize with your OpenAI API key
jarvis = create_jarvis_controller(api_key="your-api-key")

# Or use environment variable OPENAI_API_KEY
jarvis = create_jarvis_controller()
```

### Working with Individual Agents

```python
# Get a specific agent
project_agent = jarvis.get_agent('project_manager')
content_agent = jarvis.get_agent('content_creator')

# List all available agents
all_agents = jarvis.list_agents()
print(all_agents)

# Get agent information
info = jarvis.get_agent_info('financial_analyst')
print(info['goal'])
```

### Agent Information

```python
# Get details about all agents
all_info = jarvis.get_all_agents_info()

# Display agent portfolio
for name, info in all_info.items():
    print(f"{info['role']}: {info['goal']}")
```

## 🔧 Integration with Main JARVIS App

The controller can be integrated with your main JARVIS application by importing and using specific agents based on user requests:

```python
from jarvis_controller import create_jarvis_controller

# In your main app
jarvis_controller = create_jarvis_controller()

# Route requests to appropriate agents
if "project plan" in user_input:
    agent = jarvis_controller.get_agent('project_manager')
elif "content" in user_input or "blog" in user_input:
    agent = jarvis_controller.get_agent('content_creator')
elif "financial" in user_input or "budget" in user_input:
    agent = jarvis_controller.get_agent('financial_analyst')
# ... etc
```

## 📋 Key Features

- **9 Specialized Agents**: Each with unique expertise and detailed backstories
- **Sequential Processing**: Crew uses sequential process for coordinated workflows
- **Memory & Planning**: Enhanced with memory and planning capabilities
- **Flexible Access**: Easy agent retrieval and information access
- **Comprehensive Coverage**: Business, personal, creative, and technical domains

## 🔑 Requirements

- OpenAI API key (set as environment variable `OPENAI_API_KEY` or pass directly)
- CrewAI framework
- Python 3.8+

## 🎯 Next Steps

1. **Add Tasks**: Create specific tasks for agents to execute
2. **Workflow Design**: Design multi-agent workflows for complex projects
3. **Integration**: Integrate with your main JARVIS application
4. **Customization**: Modify agent backstories and goals for your specific needs

## 💡 Tips

- Each agent has detailed backstories that inform their responses
- Use the appropriate agent for the domain-specific expertise you need
- The sequential process ensures coordinated execution when using multiple agents
- Agent information can help you choose the right specialist for each task

## 🧪 Testing

Run the test suite to validate your setup:

```bash
python test_jarvis_controller.py
```

This will test:
- Controller initialization
- Agent structure and information
- Crew configuration
- Complete agent portfolio display