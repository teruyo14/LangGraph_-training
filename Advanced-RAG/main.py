from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from graph.graph import app


question = "What are the types of agent memory?"
inputs = {"question": question}

for output in app.stream(inputs, config={"configurable": {"thread_id": "2"}}):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
