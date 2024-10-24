import os
import re
import time
import json

from dotenv import load_dotenv
from pydantic import BaseModel
import anthropic

from fi_dialogue_agents import Agent

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

host = os.getenv('HOST', '127.0.0.1')
port = int(os.getenv('PORT', 5380))

claude_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

AGENT_WPM = 70

# Function to count syllables for readability scoring
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    num_vowels = 0
    prev_char_was_vowel = False
    
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                num_vowels += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    if word.endswith("e"):
        if num_vowels > 1:
            num_vowels -= 1
    
    return num_vowels

# Function to calculate readability score (for thinking time)
def readability_score(message):
    words = message.split()
    num_words = len(words)
    num_chars = sum(len(word) for word in words)
    num_syllables = sum(count_syllables(word) for word in words)
    
    if num_words == 0 or num_chars == 0:
        return 0

    score = 206.835 - 1.015 * (num_words / num_chars) - 84.6 * (num_syllables / num_words)
    return score

# Calculate thinking time based on readability score
def calculate_thinking_time(message):
    score = readability_score(message)
    if score > 90:
        return 0.5
    elif score > 80:
        return 1
    elif score > 70:
        return 1.5
    elif score > 60:
        return 2
    else:
        return 3

# Calculate typing time based on word count
def calculate_typing_time(message):
    words = message.split()
    seconds = len(words) / AGENT_WPM * 60
    return seconds

# Pydantic model for expected Claude response
class WantedResponseFormat(BaseModel):
    response: str
    thinking_time: float
    typing_time: float

# Claude response generation with metadata (multiple responses allowed)
def get_claude_response_with_metadata(message):
    start_time = time.time()

    prompt = f"""
Respond to the following message naturally, as if in casual conversation. Provide a single response, but you can follow up or clarify if necessary. Avoid overly formal language and aim for a friendly, conversational tone.

Message: '{message}'

Return a JSON object with this structure:
{{
  "response": "<your full response (with potential follow-up)>",
  "thinking_time": <seconds>,
  "typing_time": <seconds>
}}.
"""

    # Make the API call to Claude
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        system="You are a helpful, friendly conversational assistant.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    end_time = time.time()
    network_latency = end_time - start_time

    try:
        # Extract the response content
        raw_content = response.content[0].text
        print("Raw content from Claude:", raw_content)

        # Parse the JSON object directly (no need to search for '[')
        response_data = json.loads(raw_content)

        # Adjust thinking time for network latency
        response_data["thinking_time"] = max(0, response_data["thinking_time"] - network_latency)

        return [response_data]  # Return as a list to keep handling consistent
    except Exception as e:
        print(f"Error parsing response: {e}")
        return [{
            "response": "Sorry, I encountered an issue.",
            "thinking_time": max(0, 1.0 - network_latency),
            "typing_time": len(message.split()) / 70 * 60
        }]


# Function to process incoming messages
def received_message(message):
    print(f"Processing message: {message}")

    # Step 1: Generate Claude response(s)
    claude_responses = get_claude_response_with_metadata(message)
    
    for response_data in claude_responses:
        # Step 2: Calculate additional thinking time based on readability (for more human-like pacing)
        additional_thinking_time = calculate_thinking_time(message)
        time.sleep(response_data["thinking_time"] + additional_thinking_time)

        # Step 3: Send the Claude response
        response = response_data["response"]
        typing_time = response_data["typing_time"]
        
        agent.start_typing()
        time.sleep(typing_time)
        agent.send_message(response)
        agent.stop_typing()
        
# Initialize and run the agent
agent = Agent(host=host, port=port)
agent.on_message(received_message)
agent.run()
