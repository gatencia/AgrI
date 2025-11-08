"""
REACT (Reasoning and Acting) Agent Implementation
An AI agent that alternates between reasoning and acting to solve tasks.
"""

import json
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import requests


@dataclass
class Observation:
    """Represents an observation from an action."""
    result: str
    success: bool = True


@dataclass
class Action:
    """Represents an action to be taken."""
    tool_name: str
    arguments: Dict[str, Any]


class Tool:
    """Base class for tools that the agent can use."""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, **kwargs) -> Observation:
        """Execute the tool with given arguments."""
        try:
            result = self.func(**kwargs)
            return Observation(result=str(result), success=True)
        except Exception as e:
            return Observation(result=f"Error: {str(e)}", success=False)


class REACTAgent:
    """
    A REACT agent that uses reasoning and acting to solve tasks.
    
    The agent follows this loop:
    1. Think: Reason about the current situation
    2. Act: Decide on an action to take
    3. Observe: See the result of the action
    4. Repeat until task is complete
    """
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = 10
        self.iteration_count = 0
        self.observation_history: List[str] = []
        
    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenRouter API to get LLM response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def _extract_action(self, text: str) -> Optional[Action]:
        """Extract action from LLM response."""
        # Look for action pattern: Action: tool_name(arg1=value1, arg2=value2)
        action_pattern = r'Action:\s*(\w+)\s*\(([^)]*)\)'
        match = re.search(action_pattern, text)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        arguments = {}
        if args_str.strip():
            # Simple argument parsing (key=value pairs)
            arg_pattern = r'(\w+)=([^,]+)'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2).strip().strip('"\'')
                # Try to parse as number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                arguments[key] = value
        
        return Action(tool_name=tool_name, arguments=arguments)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_description = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""You are an agent that is helping a user analyze their farm with satellite data

Available Tools:
{tools_description}

Your task is to:
1. Think about the current situation and what data you need to gather. Then, call the correct tools.
2. Use the format: Action: tool_name(arg1=value1, arg2=value2)
3. After observing results, think again and decide on the next action
4. When the task is complete, respond with: Final Answer: [your answer]

Format your responses as:
Thought: [your reasoning]
Action: [tool_name(arguments)] or Final Answer: [answer]"""
    
    def _execute_action(self, action: Action) -> Observation:
        """Execute an action using the appropriate tool."""
        if action.tool_name not in self.tools:
            return Observation(
                result=f"Tool '{action.tool_name}' not found. Available tools: {', '.join(self.tools.keys())}",
                success=False
            )
        
        tool = self.tools[action.tool_name]
        return tool.execute(**action.arguments)
    
    def run(self, task: str) -> str:
        """
        Run the REACT loop to solve a task.
        
        Args:
            task: The task description
            
        Returns:
            Final answer or result
        """
        self.iteration_count = 0
        self.observation_history = []
        
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            
            # Get LLM response (think + act)
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response})
            
            # Check for final answer
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()
            
            # Extract and execute action
            action = self._extract_action(response)
            if not action:
                # If no action found, might be just thinking
                if "Thought:" in response:
                    continue
                else:
                    return f"Could not extract action. Response: {response}"
            
            # Execute action
            observation = self._execute_action(action)
            self.observation_history.append(f"Iteration {self.iteration_count}: {action.tool_name} -> {observation.result}")
            
            # Add observation to messages
            observation_text = f"Observation: {observation.result}"
            messages.append({"role": "user", "content": observation_text})
        
        return f"Reached maximum iterations ({self.max_iterations}). Last observations: {'; '.join(self.observation_history[-3:])}"


def get_weather(location: str) -> str:
    """Placeholder weather function. Replace with actual weather API integration."""
    return f"Weather data for {location} would be retrieved here. (Placeholder)"


class AgriAgent(REACTAgent):
    """
    An agent that is specialized in agriculture.
    """
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        super().__init__(api_key, model)
        self.tools = [
            Tool(name="get_weather", description="Get the weather for a given location", func=get_weather)
        ]
        self.bounding_points = []
    
    def add_bounding_point(self, point: tuple[float, float]):
        """Add a bounding point to the agent."""
        self.bounding_points.append(point)

def main():
    """Example usage of the REACT agent."""
    # Load API key
    try:
        with open("OPENROUTER_API_KEY.txt", "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        print("Error: OPENROUTER_API_KEY.txt not found")
        return
    
    # Create agent
    agent = REACTAgent(api_key=api_key)
    
    
    
    # Example task
    task = "Calculate 15 * 23"
    
    print(f"Task: {task}\n")
    print("Running REACT agent...\n")
    
    result = agent.run(task)
    
    print(f"\nFinal Result: {result}")
    print(f"\nIterations used: {agent.iteration_count}")
    print(f"\nObservation History:")
    for obs in agent.observation_history:
        print(f"  - {obs}")


if __name__ == "__main__":
    main()

