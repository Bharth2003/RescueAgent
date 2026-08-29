from strands import Agent, tool
import datetime
@tool
def get_current_time(timezone: str)-> str:
    """Get the current time in a specified timezone. Always use this to check the timezone."""
    return f"The time is exactly {datetime.datetime.now().strftime('%H:%M')}"
my_agent = Agent(tools=[get_current_time])
print("Asking the agent...")
result = my_agent("What is the time right now?")
print(f"\nAgent says: {result.message}")
