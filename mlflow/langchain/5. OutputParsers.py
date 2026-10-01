import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, ChatMessagePromptTemplate
from langchain_core.output_parsers import ListOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from langchain_core.globals import set_debug

set_debug(True)

os.environ["GOOGLE_API_KEY"] = "AQ."

output_parser = CommaSeparatedListOutputParser()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
    max_retries=5
)

prompt = PromptTemplate(
    template="""
List 3 {things}.
{format_instructions}
""",
    input_variables=["things"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

formatted_prompt = prompt.format(
    things="fruits"
)


# Call Gemini
response = llm.invoke(formatted_prompt)


# Parse output
result = output_parser.parse(
    response.content[0]["text"]
)


print(result)
print(type(result))

