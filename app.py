'''from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env file (force override just in case)
load_dotenv('./credentials/.env', override=True)

# Get the key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file!")

# Pass the key into the client
client = OpenAI(api_key=openai_key)

# Make a test request
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Say hello world"
)

print(response.output_text)
'''

