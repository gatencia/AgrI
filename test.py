import subprocess
import tempfile
import textwrap

def code_executor_agent(code: str):
    """
    Agent B: Executes Python code in a safe, temporary subprocess.
    Returns stdout and stderr as strings.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=10  # safety timeout
            )
            return result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "", "Execution timed out"


def coordinator_agent(task: str):
    """
    Agent A: Decides what code to run and delegates it to the code executor agent.
    """
    print(f"\nAgent A received task: {task}")

    if "sum" in task.lower():
        code = """
        numbers = [1, 2, 3, 4, 5]
        print("Sum:", sum(numbers))
        """
    elif "fibonacci" in task.lower():
        code = """
        def fib(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        print("Fib(10):", fib(10))
        """
    else:
        code = 'print("Task not recognized")'

    print("Agent A: Sending code to Agent B for execution...")
    stdout, stderr = code_executor_agent(code)

    print("\n--- Agent B Output ---")
    if stdout:
        print(stdout)
    if stderr:
        print("\n[stderr]")
        print(stderr)


if __name__ == "__main__":
    coordinator_agent("Compute the sum of 1 to 5")
    coordinator_agent("Compute fibonacci number 10")
    coordinator_agent("Unknown task")
