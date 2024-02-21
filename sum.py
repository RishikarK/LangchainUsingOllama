from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama2:latest", temperature=0.6, format="json")

speech = """ 
No-show for first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 29055 INR / 29055 INR (at today exchange rates 29055 INR / 29055 INR)New travel dates and change request must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
No-show for subsequent flight(s)
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 10795 INR / 10795 INR (at today exchange rates 10795 INR / 10795 INR)New travel dates and change request must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
Prior to Departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
After departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
Hyderabad - Barcelona
No-show for first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 29055 INR / 29055 INR (at today exchange rates 29055 INR / 29055 INR)New travel dates and change request must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
No-show for subsequent flight(s)
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 10795 INR / 10795 INR (at today exchange rates 10795 INR / 10795 INR)New travel dates and change request must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
Prior to Departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
After departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR" """

messages = [
     SystemMessage(
        content="You are an expert assistant with expertise in summarizing speeches"
    ),
    HumanMessage(
        content=f"Summarize this and make it concise and provide the bullet points for each subject for user interface:\n TEXT: {speech}"
    ),
]

chat_model_response = llm.invoke(messages)
print(chat_model_response.content)
