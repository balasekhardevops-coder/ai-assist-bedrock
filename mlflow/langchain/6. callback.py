import time, os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AQ."


# 1. Create Callback Handler
class TimingCallback(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start = time.time()
        print("LLM Started")
        print("Prompt:", prompts)


    def on_llm_end(self, response, **kwargs):
        end = time.time()

        print("LLM Finished")

        print(
            "Latency:",
            end - self.start,
            "seconds"
        )


# 2. Create LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# 3. Create Callback object
callback = TimingCallback()


# 4. Invoke LLM  <-- Actual LLM call happens here
response = llm.invoke(
    "Explain Kubernetes in simple terms",
    config={
        "callbacks": [callback]
    }
)


# 5. Print response
print("\nFinal Response:")
print(response.content)