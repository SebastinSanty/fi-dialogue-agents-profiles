import os
import re
import time

from dotenv import load_dotenv

from fi_dialogue_agents import Agent

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

host = os.getenv('HOST', '127.0.0.1')
port = int(os.getenv('PORT', 5380))

AGENT_WPM = 70

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

def readability_score(message):
    words = message.split()
    num_words = len(words)
    num_chars = sum(len(word) for word in words)

    num_syllables = sum(count_syllables(word) for word in words)
    
    if num_words == 0 or num_chars == 0:
        return 0

    score = 206.835 - 1.015 * (num_words / num_chars) - 84.6 * (num_syllables / num_words)
    return score

def thinking_time(message):
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

def calculate_time_from_wpm(message):
    words = message.split()
    seconds = len(words) / AGENT_WPM * 60
    return seconds

def received_message(message):
    print(f"Processing message: {message}")
    time.sleep(thinking_time(message))  
    agent.start_typing()
    time.sleep(calculate_time_from_wpm(message))
    agent.send_message(message)
    agent.stop_typing()

    last_word = message.split()[-1]
    reversed_last_word = last_word[::-1].lower()

    message_2 = f"Or should I say {reversed_last_word}"
    agent.start_typing()
    time.sleep(calculate_time_from_wpm(message_2))
    agent.send_message(message_2)
    agent.stop_typing()

agent = Agent(host=host, port=port)
agent.on_message(received_message)
agent.run()
