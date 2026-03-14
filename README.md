# agentic-day4-multi-agent
Multi-Agent Collaboration: Supervisor + Specialists

## Objective

Build a **multi-agent customer support system** that applies the patterns from **Week 2 Session 4 – Multi-Agent Collaboration**:

- A **supervisor agent** that routes requests
- **Specialist agents** for different domains
- **Structured handoffs** between agents
- **Graceful degradation** (4-level fallback)
- **Session-level audit log with cost tracking**

## Project Structure
```
agentic-day4-multi-agent/
├── .gitignore
├── requirements.txt
├── README.md
├── app.py
└── prompts/
    └── supervisor_v1.yaml
```

## How to run the App 
1. Clone the Repository:
```
    git clone https://github.com/bipulsenapati998/agentic-day4-multi-agent.git
```
2. Create & Activate the virtual environment:
```
    Use conda	
    conda create -n llms python=3.11 && conda activate llms  
```
3. Update the .env file with your OpenAI API key:
```
    OPENAI_API_KEY=<your_actual_api_key_here>
```
4. Install Dependencies:
```
    pip install -r requirement.txt
```
5. Run the Application:
```
   python app.py 
```