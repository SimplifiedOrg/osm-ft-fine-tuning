"""
test_finetuned_model.py
"""
import json

import ollama

MODEL_NAME = "tx-categorizer"

UNSEEN_EXAMPLES = [
    {"description": "POS 3312 LIME SCOOTER SAN FRANCISCO CA", "amount": -4.50},
    {"description": "AUTOPAY PEACOCK TV", "amount": -7.99},
    {"description": "AUTOPAY ORANGETHEORY FITNESS", "amount": -159.00},
    {"description": "DIRECT DEP GUSTO PAYROLL", "amount": 2450.00},
    {"description": "CASH APP TRANSFER TO M GARCIA", "amount": -120.00},
    {"description": "ATM WITHDRAWAL PENDING REVERSAL 5521 CHASE NYC", "amount": -500.00},
    {"description": "BEST BUY 45TH ST DISPUTE PENDING", "amount": -899.00},
    {"description": "AUTOPAY VERIZON WIRELESS", "amount": -85.00},
    {"description": "POS 7734 TRADER JOES SAN DIEGO CA", "amount": -64.13},
    {"description": "CIRCLE K GAS DENVER CO", "amount": -52.30},
    {"description": "ALAMO DRAFTHOUSE CINEMA AUSTIN TX", "amount": -18.50},
]

def categorize(example):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": json.dumps(example)},
        ],
    )
    return response["message"]["content"]

if __name__ == "__main__":
    for example in UNSEEN_EXAMPLES:
        result = categorize(example)
        print(f"{example['description']}  ({example['amount']})")
        print(f"  -> {result}\n")