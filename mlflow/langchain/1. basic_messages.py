import os

from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = "AQ"


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)


text = "What would be a good company name for a company that makes toys for kids?"


response = llm.invoke(text)


print(response.content[0]["text"])