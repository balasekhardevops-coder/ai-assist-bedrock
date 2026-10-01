import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)


# Set API Key
os.environ["GOOGLE_API_KEY"] = "AQ."


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3
)


# Few-shot examples
examples = [
    {
        "review": "This phone is amazing. I love the camera.",
        "sentiment": "Positive"
    },
    {
        "review": "The product broke after two days. Very disappointed.",
        "sentiment": "Negative"
    },
    {
        "review": "The laptop is okay, nothing special.",
        "sentiment": "Neutral"
    }
]


# Template for examples
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{review}"),
        ("ai", "{sentiment}")
    ]
)


# Create few-shot message template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)


# Final prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sentiment classification expert. "
            "Answer only Positive, Negative, or Neutral."
        ),

        few_shot_prompt,

        (
            "human",
            "{review}"
        )
    ]
)


# Create messages
messages = final_prompt.format_messages(
    review="The battery performance is excellent and I really like this device."
)


# Call Gemini
response = llm.invoke(messages)


# Print clean output
print(response.content[0]["text"])