from control_flow import app

# Test on current events
inputs = {
    "question": "What are the models for llama3.2?",
    "max_retries": 3,
}

for event in app.stream(inputs, stream_mode="values"):
    print(event)
