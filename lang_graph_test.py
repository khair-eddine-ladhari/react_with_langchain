import os
import pandas as pd
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()





llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)




@tool
def calculate_dosage(weight_kg: float, mg_per_kg: float) -> str:
    """Calculate a weight-based drug dosage in mg."""
    dose = weight_kg * mg_per_kg
    return f"Recommended dose: {dose:.1f} mg"

@tool
def retrieve_patient_info(name: str) -> str:
    """Retrieve patient information based on their name."""
    patients = pd.read_csv("patients.csv")
    patient_info = patients[patients["name"] == name]
    if patient_info.empty:
        return f"No patient found with name {name}"
    return patient_info.to_string()

@tool
def department_report(department: str, revenue: int, expenses: int) -> str:
    """Generate a financial report for a hospital department."""
    net = revenue - expenses
    return (
        f"Report for {department}:\n"
        f"Revenue: ${revenue}\nExpenses: ${expenses}\nNet: ${net}"
    )




tools = [calculate_dosage, retrieve_patient_info, department_report]
agent = create_react_agent(llm, tools)





messages = agent.invoke({
    "messages": [("human", "Calculate the dosage for a 70kg patient at 5mg per kg")]
})
print(messages["messages"][-1].content)

messages = agent.invoke({
    "messages": [("human", "Create a summary for patient John Doe")]
})
print(messages["messages"][-1].content)