# REACT Agent

A small AI agent implementation using the REACT (Reasoning and Acting) loop pattern. The agent alternates between reasoning about a task and taking actions to solve it.

## Features

- **REACT Loop**: Implements the reasoning-action-observation cycle
- **Tool System**: Extensible tool/action framework
- **OpenRouter Integration**: Uses OpenRouter API for LLM reasoning
- **Built-in Tools**: Calculator, search (simulated), and memory storage

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `OPENROUTER_API_KEY.txt` file contains your OpenRouter API key

## Usage

### Basic Usage

Run the example:
```bash
python orchestrator.py
```

### Custom Usage

```python
from orchestrator import REACTAgent, create_calculator_tool, create_memory_tool

# Load API key
with open("OPENROUTER_API_KEY.txt", "r") as f:
    api_key = f.read().strip()

# Create agent
agent = REACTAgent(api_key=api_key, model="openai/gpt-4o-mini")

# Register tools
agent.register_tool(create_calculator_tool())
agent.register_tool(create_memory_tool())

# Run a task
result = agent.run("Calculate 10 * 5 and store the result")
print(result)
```

## How It Works

1. **Think**: The agent reasons about the current situation using an LLM
2. **Act**: The agent decides on an action (using a tool)
3. **Observe**: The agent sees the result of the action
4. **Repeat**: The cycle continues until the task is complete

The agent uses a structured format:
- `Thought: [reasoning]`
- `Action: tool_name(arg1=value1, arg2=value2)`
- `Final Answer: [result]`

## Available Tools

- **calculator**: Evaluates mathematical expressions
- **search**: Simulated web search (can be extended with real API)
- **memory**: Store and retrieve information

## Creating Custom Tools

```python
from orchestrator import Tool

def my_custom_function(param1: str, param2: int) -> str:
    # Your tool logic here
    return f"Result: {param1} {param2}"

custom_tool = Tool(
    name="my_tool",
    description="Does something useful. Usage: my_tool(param1='value', param2=123)",
    func=my_custom_function
)

agent.register_tool(custom_tool)
```

## Configuration

- `max_iterations`: Maximum number of REACT loop iterations (default: 10)
- `model`: OpenRouter model to use (default: "openai/gpt-4o-mini")

