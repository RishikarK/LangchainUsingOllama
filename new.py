from transformers import AutoTokenizer, AutoModelForCausalLM

def summarize_text_with_ollama(text):
    # Load Ollama model and tokenizer
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage
extracted_text = """No-show for first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 29055 INR / 29055 INR (at today exchange rates 29055 INR / 29055 INR)New travel dates and change request must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
No-show for subsequent flight(s)
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 10795 INR / 10795 INR (at today exchange rates 10795 INR / 10795 INR)New travel dates and change request must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
Prior to Departure of first flight
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 I NR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Thursday, February 06, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR
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
Changes*: Not applicable (See reissue conditions)Reissue: Allowed with restrictionsPenalty fee for ticket reissue between: 0 INR / 10795 INR (at today exchange rates 0 INR / 10795 INR)New travel dates must be prior to: Wednesday, February 19, 2025Maximum Reissue penalty fee for entire ticket: 29055 INR" 
""" # Your extracted text here
summary = summarize_text_with_ollama(extracted_text)
print(f"Summary:\n{summary}")

