from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pathlib import Path
import json
import yaml

# ------------------------------------------------------------
#   Load environment variables & define constants
# ------------------------------------------------------------
load_dotenv()
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
BUDGET_USD = 0.50
PROMPT_PATH = Path("./prompts/supervisor_v1.yaml")
VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}

llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)


# ------------------------------------------------------------
#   State
# ------------------------------------------------------------
class MultiAgentState(TypedDict):
    user_request: str  # original user message
    route: str  # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str  # which specialist handled it
    specialist_result: str  # raw output from specialist agent
    final_response: str  # final response returned to the user


# ------------------------------------------------------------
#   Load Supervisor prompt
# ------------------------------------------------------------
def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise ValueError(f"Prompt not found: {PROMPT_PATH}")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load()
    return data["system"].strip()


SUPERVISOR_SYSTEM_PROMPT = load_prompt()


# ------------------------------------------------------------
#   Supervisor Node & Routing
# ------------------------------------------------------------
def supervisor_node(state: MultiAgentState) -> dict:
    """Calls LLM with YAML prompt, sets the route."""
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    return {"route": route}


def route_to_specialist(state: MultiAgentState) -> str:
    """Returns the node name based on the route."""
    route_map = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")


# ------------------------------------------------------------
#   Specialist Agent
# ------------------------------------------------------------
def orders_agent_node(state: MultiAgentState) -> dict:
    text = f"[orders_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "orders_agent",
        "specialist_result": text,
    }


def billing_agent_node(state: MultiAgentState) -> dict:
    text = f"[billing_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "billing_agent",
        "specialist_result": text,
    }


def technical_agent_node(state: MultiAgentState) -> dict:
    text = f"[technical_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "technical_agent",
        "specialist_result": text,
    }


def subscription_agent_node(state: MultiAgentState) -> dict:
    text = f"[subscription_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "subscription_agent",
        "specialist_result": text,
    }


def general_agent_node(state: MultiAgentState) -> dict:
    text = f"[general_agent] Handling request: {state['user_request']}"
    return {
        "agent_used": "general_agent",
        "specialist_result": text,
    }


def synthesize_response_node(state: MultiAgentState) -> dict:
    """Combines specialist result into a final response."""
    return {"final_response": state["specialist_result"]}


# ------------------------------------------------------------
#   Build the Multi‑Agent Graph
# ------------------------------------------------------------
def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")


# ------------------------------------------------------------
#   MAIN DEMONSTRATION
# ------------------------------------------------------------


def main() -> None:
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    for request in [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
    ]:
        safe_text = guard_request(request)
        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
        }
        result = graph.invoke(state)
        print("Request:", request)
        print("Route:", result.get("route"), "Agent used:", result.get("agent_used"))
        print("Final:", result.get("final_response"))
        print("---")

    print("Total cost (USD):", round(audit.total_cost_usd, 6))
    persist_audit_log(audit)


if __name__ == "__main__":
    main()
