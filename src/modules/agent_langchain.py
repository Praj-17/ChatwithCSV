from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import json
load_dotenv()


class ChatwithCSV:
    def __init__(self, openai_key) -> None:
        self.openai_key = openai_key
        self.llm = ChatOpenAI(
            temperature=0, model="gpt-4o-mini", openai_api_key=openai_key, streaming=True
        )
    def chat_with_a_df(self, df, question):
        print(df, question)
        agent_executor = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True,
            prefix="You are an excellent data analyst who can answers questions based on a given pandas dataframe if you can not figure out the answer just politely say `The given context does not provide answer to the following problem`",
        )

        ans = agent_executor.invoke({"input": question})
        print(ans)
        return ans.get("output", "I dont know")

if __name__ == "__main__":
    openai_key = ""
    chat = ChatwithCSV(openai_key)
    df = pd.read_csv(r"src\data\vitals.csv")
    samples = []
    while True:
        queestion = input("Please enter your question here: ")
        if queestion == "exit":
            break
        ans = chat.chat_with_a_df(df, question= queestion)
        print(ans)

        samples.append({"question" : queestion, "answer": ans})



    with open("sample_queries.json", "w") as f:
        json.dump(samples, f, indent=4)
    
    print("Files Saved")

        

