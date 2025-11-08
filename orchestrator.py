"""
REACT (Reasoning and Acting) Agent Implementation
An AI agent that alternates between reasoning and acting to solve tasks.
"""

import json
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import requests
import subprocess
import tempfile
import textwrap
import os


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
    
    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2-thinking"):
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
        action_pattern = r'Action:\s*(\w+)\s*\('
        match = re.search(action_pattern, text, re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        # Find the opening parenthesis position
        start_pos = match.end()
        
        # Find the matching closing parenthesis by counting nested parentheses
        paren_count = 1
        i = start_pos
        in_string = False
        string_char = None
        escape_next = False
        
        while i < len(text) and paren_count > 0:
            if escape_next:
                escape_next = False
            elif text[i] == '\\' and in_string:
                escape_next = True
            elif text[i] in ['"', "'"] and not escape_next:
                if not in_string:
                    in_string = True
                    string_char = text[i]
                elif text[i] == string_char:
                    # Check for triple quotes
                    if i + 2 < len(text) and text[i:i+3] == string_char * 3:
                        i += 2  # Skip the next two quote chars
                    else:
                        in_string = False
                        string_char = None
            elif not in_string:
                if text[i] == '(':
                    paren_count += 1
                elif text[i] == ')':
                    paren_count -= 1
            
            if paren_count > 0:
                i += 1
        
        if paren_count != 0:
            return None
        
        # Extract arguments string (everything between parentheses)
        args_str = text[start_pos:i]
        
        # Parse arguments with proper quote handling
        arguments = {}
        if args_str.strip():
            # More robust argument parsing that handles quoted strings
            i = 0
            while i < len(args_str):
                # Skip whitespace
                while i < len(args_str) and args_str[i] in ' \n\t':
                    i += 1
                if i >= len(args_str):
                    break
                
                # Find key
                key_match = re.match(r'(\w+)\s*=\s*', args_str[i:])
                if not key_match:
                    break
                
                key = key_match.group(1)
                i += len(key_match.group(0))
                
                # Find value
                if i >= len(args_str):
                    break
                
                # Check if value starts with a quote
                if args_str[i] in ['"', "'"]:
                    quote_char = args_str[i]
                    i += 1
                    
                    # Check for triple quotes
                    is_triple_quote = False
                    if i + 1 < len(args_str) and args_str[i:i+2] == quote_char * 2:
                        is_triple_quote = True
                        quote_char = quote_char * 3
                        i += 2
                    
                    value_start = i
                    value = ""
                    
                    # Find matching quote (handling escaped quotes)
                    while i < len(args_str):
                        if is_triple_quote:
                            # For triple quotes, look for three consecutive quote chars
                            if i + 2 < len(args_str) and args_str[i:i+3] == quote_char:
                                i += 3
                                break
                            elif args_str[i] == '\\' and i + 1 < len(args_str):
                                value += args_str[i:i+2]
                                i += 2
                            else:
                                value += args_str[i]
                                i += 1
                        else:
                            # For single/double quotes
                            if args_str[i:i+len(quote_char)] == quote_char:
                                i += len(quote_char)
                                break
                            elif args_str[i] == '\\' and i + 1 < len(args_str):
                                value += args_str[i:i+2]
                                i += 2
                            else:
                                value += args_str[i]
                                i += 1
                    
                    arguments[key] = value
                else:
                    # Unquoted value - find until comma or end
                    value_start = i
                    paren_count = 0
                    bracket_count = 0
                    brace_count = 0
                    
                    while i < len(args_str):
                        if args_str[i] == '(':
                            paren_count += 1
                        elif args_str[i] == ')':
                            paren_count -= 1
                        elif args_str[i] == '[':
                            bracket_count += 1
                        elif args_str[i] == ']':
                            bracket_count -= 1
                        elif args_str[i] == '{':
                            brace_count += 1
                        elif args_str[i] == '}':
                            brace_count -= 1
                        elif args_str[i] == ',' and paren_count == 0 and bracket_count == 0 and brace_count == 0:
                            break
                        i += 1
                    
                    value = args_str[value_start:i].strip()
                    
                    # Try to parse as number if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
                    
                    arguments[key] = value
                
                # Skip comma
                while i < len(args_str) and args_str[i] in ', \n\t':
                    i += 1
        
        return Action(tool_name=tool_name, arguments=arguments)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_description = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        print(f"Tools description: {tools_description}")
        return f"""You are an agent that is helping a user analyze their farm with satellite data. Use the python interpreter to analyze the satellite imagery and \
            then aggregate the data into a recommendation.

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



def execute_python_code(code: str) -> str:
    """
    Execute Python code in a safe, temporary subprocess.
    Returns stdout and stderr as a formatted string.
    """
    # Decode escape sequences (e.g., \n -> actual newline, \\ -> \, etc.)
    # The code string may contain literal escape sequences that need to be converted
    try:
        # Use bytes decode with unicode_escape to properly handle all escape sequences
        # This converts literal \n to actual newlines, \\ to \, etc.
        code = code.encode('latin-1').decode('unicode_escape')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback: manually replace common escape sequences
        # First protect escaped backslashes
        code = code.replace('\\\\', '\x00BS\x00')
        # Replace escape sequences
        code = code.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        code = code.replace('\\"', '"').replace("\\'", "'")
        # Restore escaped backslashes
        code = code.replace('\x00BS\x00', '\\')
    except Exception:
        # If all else fails, use code as-is
        pass
    
    print(f"Executing code: {code[:100]}...")  # Print first 100 chars to avoid spam
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        temp_file_path = f.name
    
    try:
        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10  # safety timeout
        )
        
        output_parts = []
        if result.stdout.strip():
            output_parts.append(f"Output:\n{result.stdout.strip()}")
        if result.stderr.strip():
            output_parts.append(f"Errors:\n{result.stderr.strip()}")
        if result.returncode != 0:
            output_parts.append(f"Return code: {result.returncode}")
        
        return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)"
    
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 10 seconds - fix code and try again"
    except Exception as e:
        return f"Error executing code: {str(e)} - fix and try again"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

def get_ndvi():
    """Get the NDVI of the bounding box."""
    return "NDVI: 0.5"
class AgriAgent(REACTAgent):
    """
    An agent that is specialized in agriculture.
    """
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        super().__init__(api_key, model)
        # Register tools using the proper method
        
        self.register_tool(Tool(
            name="execute_code",
            description="Execute Python code in a safe subprocess. Returns stdout and stderr. Usage: execute_code(code='print(\"def fib(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        a, b = b, a + b\\n    return a\\nprint(\"Fib(10):\", fib(10))')",
            func=execute_python_code
        ))
        self.bounding_points = []
    
    def add_bounding_point(self, point: tuple[float, float]):
        """Add a bounding point to the agent."""
        self.bounding_points.append(point)
    def get_square_bounding_box(self) -> tuple[float, float, float, float]:
        """Get the square bounding box of the agent."""
        if len(self.bounding_points) < 2:
            return None
        xs, ys = zip(*self.bounding_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

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
    agent = AgriAgent(api_key=api_key)
    
    
    
    # Example task
    task = "Write and test a python script that reads the file README.md and returns the number of lines in the file"
    
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

