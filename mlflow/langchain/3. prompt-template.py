import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, ChatMessagePromptTemplate


os.environ["GOOGLE_API_KEY"] = "AQ."

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_retries=5
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}."),
        ("human", "Explain {topic}.")
    ]
)

messages = prompt.format_messages(
    role="Andhra Pradesh Chief Minister",
    topic="Andhra Pradesh's new policies for the year 2024"
)

response = llm.invoke(messages)
print(response.content[0]["text"])

