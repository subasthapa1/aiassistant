import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load .env file using relative path
dotenv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'credentials', '.env')
)
load_dotenv(dotenv_path)  # <-- You missed this in your code

# Get the key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# Tool creation
tools = [multiply]

# Create model with API key
model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

# Bind tools to the model
model_with_tools = model.bind_tools(tools)

# User input
user_input = "Multiply 2 and 3"

# Invoke the model
response = model_with_tools.invoke(user_input)

# Output tool calls
print(response.tool_calls)
