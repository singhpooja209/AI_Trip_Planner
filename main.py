from fastapi import FastAPI
from pydantic import BaseModel
from json import JSONResponse
from agents.agentic_workflow import GraphBuilder
import os
app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    # user_id: str = None
    # session_id: str = None
    # context: dict = None
    # model_provider: str = "groq"  # Default to Groq, can be overridden by user input

@app.get("/query")

async def query_travel_agent(query: QueryRequest):
    try:
        print(f"Received query: {query.query}")
        graph = GraphBuilder(model_provider= "groq")
        react_app = graph()

        png_graph = react_app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_graph)
        
        print(f"graph save as graph.png in {os.getcwd()}")
        messages = {"messages":[query.question]}
        output = react_app.invoke(messages)
        if isinstance(output, dict) and "messages" in output:
            final_oupt = output["messages"][-1].content
            print(f"Final output: {final_oupt}")

        else:
            final_output = str(output)
            print(f"Final output: {final_output}")
    except Exception as e:
        print(f"Error processing query: {e}")
        return JSONResponse(
            {"error": "An error occurred while processing your query."},
            status=500
        )