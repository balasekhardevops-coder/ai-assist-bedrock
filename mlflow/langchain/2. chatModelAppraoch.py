import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AQ."

system_message = SystemMessage(content="You are a creative branding expert who creates premium toy company names for kids.")

human_message = HumanMessage(content="What would be a good company name for a company that makes toys for kids?")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", 
    temperature=0.7,
    max_retries=5
)

messages = [
system_message, human_message
]

response = llm.invoke(messages)
print(response.content[0]["text"])
