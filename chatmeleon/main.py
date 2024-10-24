import os
import re
import time

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

from fi_dialogue_agents import Agent

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class WantedResponseFormat(BaseModel):
    response: str
    thinking_time: float
    typing_time: float

def get_chatgpt_response_with_metadata(message):
    start_time = time.time()

    messages = [
        {
            "role": "system",
            "content": "You are a friendly and playful assistant. Keep the conversation light, casual, and fun. Use a conversational tone, sprinkle in humor when appropriate, and feel free to be creative with your responses. Stay positive and approachable while still being helpful, and make sure the user feels relaxed and entertained during the chat. Avoid being overly formal or serious—think of it like chatting with a fun, supportive friend! Respond to the user's message and provide the estimated thinking time and typing time in seconds."
        },
        {
            "role": "user",
            "content": f"Respond to the following message and provide the thinking time and typing time.\n\nMessage: '{message}'"
        }
    ]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=WantedResponseFormat,
    )
    
    end_time = time.time()
    network_latency = end_time - start_time

    try:
        print(completion.choices[0].message)
        response_message = completion.choices[0].message.content
        response_data = eval(response_message)

        response_data["thinking_time"] = max(0, response_data["thinking_time"] - network_latency)
        print(response_data)

    except Exception as e:
        print(f"Error parsing response: {e}")
        response_data = {
            "response": "Sorry, I encountered an issue.",
            "thinking_time": max(0, 1.0 - network_latency),
            "typing_time": len(message.split()) / 70 * 60
        }
    
    return response_data

def received_message(message):
    print(f"Processing message: {message}")

    chatgpt_data = get_chatgpt_response_with_metadata(message)
    
    thinking_time = chatgpt_data["thinking_time"]
    time.sleep(thinking_time)

    response = chatgpt_data["response"]
    typing_time = chatgpt_data["typing_time"]
    
    agent.start_typing()
    time.sleep(typing_time)
    agent.send_message(response)
    agent.stop_typing()

agent = Agent(host="127.0.0.1", port=5381)
agent.on_message(received_message)
agent.run()
