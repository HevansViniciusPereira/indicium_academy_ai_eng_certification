import os
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Configure LLM
# Using a lower temperature for more stable tool usage
os.getenv("HAGGINGFACE_API_KEY")

llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

# 4) Narrow Policy/Prompt (Agent Behavior)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = (
    "You are a data-focused assistant. "
    "If a question requires information from the CSV, first use an appropriate tool. "
    "Use only one tool call per step if possible. "
    "Answer concisely and in a structured way. "
    "If no tool fits, briefly explain why.\n\n"
    "Available tools:\n{tools}\n"
    "Use only these tools: {tool_names}."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [tool_schema, tool_nulls, tool_describe]

_tool_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
_tool_names = ", ".join(t.name for t in tools)
prompt = prompt.partial(tools=_tool_desc, tool_names=_tool_names)

# Create and run tool-calling agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
    )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,   # True for debug logs
    max_iterations=3, # how many reasoning–tool–response cycles the agent may perform
)

if __name__ == "__main__":
    user_query = "Which columns have missing values? List 'Column: Count'."
    result = agent_executor.invoke({"input": user_query})
    print("\nAGENT ANSWER")
    print(result["output"])

def ask_agent(query: str) -> str:
    return agent_executor.invoke({"input": query})["output"]