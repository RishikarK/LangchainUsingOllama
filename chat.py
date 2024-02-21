import requests
import json
import streamlit as st

conversation_history = []

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

def generate_response(prompt):
    global conversation_history

    conversation_history.append(prompt)
    full_prompt = "\n".join(conversation_history)

    data = {
        "model": "openchat:latest",
        "stream": False,
        "prompt": full_prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        conversation_history.append(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None

prompt = st.text_area("Enter your prompt here...")
if st.button("Generate Response"):
    response = generate_response(prompt)
    st.write(response)
