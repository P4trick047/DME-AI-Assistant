# ============================================================
# src/agent.py
# LangChain ReAct Agent with DME billing tools
# The agent can decide which tool to call based on the question
# ============================================================

import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.tools import BaseTool

from src.billing_tools import HCPCSLookupTool, ClaimValidatorTool, DenialAnalyzerTool
from src.vector_store import DMEVectorStore
from config.settings import DEFAULT_MODEL, LLM_CONTEXT_WINDOW

logger = logging.getLogger(__name__)

# The ReAct prompt follows: Thought → Action → Observation → Final Answer
AGENT_PROMPT_TEMPLATE = """You are a DME medical billing expert with access to tools.

Available tools:
{tools}

Tool names: {tool_names}

Always think step by step. Use tools when specific lookups are needed.
For general billing questions, answer from your knowledge directly.

Format to follow:
Question: the question to answer
Thought: think about which tool (if any) to use
Action: the tool name (must be one of: {tool_names}) — omit if answering directly
Action Input: the input to the tool
Observation: the tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: your complete, helpful answer

Question: {input}
{agent_scratchpad}"""


def create_billing_agent(model_name: str = DEFAULT_MODEL) -> AgentExecutor:
    """
    Create a LangChain ReAct agent with DME billing tools.

    The agent automatically decides whether to:
    - Call the HCPCS lookup tool (for code queries)
    - Call the claim validator (for validation requests)
    - Call the denial analyzer (for denial questions)
    - Answer directly from its training knowledge
    """
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,
        num_predict=1024,
        num_ctx=LLM_CONTEXT_WINDOW,
    )

    tools = [
        HCPCSLookupTool(),
        ClaimValidatorTool(),
        DenialAnalyzerTool(),
    ]

    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=AGENT_PROMPT_TEMPLATE,
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    logger.info(f"Billing agent created with model: {model_name}")
    return executor
