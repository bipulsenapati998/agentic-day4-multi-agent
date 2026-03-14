from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#   State
# -------------------------------------------------------------------
class MultiAgentState(TypedDict):
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user