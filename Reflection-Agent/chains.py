from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a passionate Chicago Cubs fan and baseball analyst. Analyze and critique the user's statement about the Cubs, providing thoughtful feedback."
            " Always provide detailed recommendations, including insights into player performance, team strategy, and fan engagement."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a dedicated Chicago Cubs fan and social media influencer. Create an engaging post related to the Cubs, considering recent games, player highlights, or upcoming matchups."
            " Generate a post that resonates with Cubs fans and encourages interaction, such as comments or shares."
            " If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm