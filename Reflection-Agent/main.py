from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


GENERATE = "generate"
REFLECT = "reflect"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 5:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    inputs = HumanMessage(content="""Make this tweet better about the Chicago Cubs:
                                    The Cubs are on fire this season! 🔥
            With incredible performances by the pitching staff and clutch hitting, they are looking like serious contenders.

            Can't wait to see how they fare in the upcoming series against their rivals!

            Let's go Cubbies! #Cubs
                                  """)
    response = graph.invoke(inputs)

