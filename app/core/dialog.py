# DialogState extracted from server.py
import re, os
from app.config import *

class DialogState:
    def __init__(self):
        self.vehicle = False
        self.location = False
        self.issue = False
        self.urgency = False
        self.first_reply_sent = False
        self.prompt_rephrase_ix = 0

    def update_from_text(self, text: str):
        t = text.lower()
        if re.search(r"\b(20\d{2}|19\d{2})\b", t) or re.search(VEHICLE_MAKES, t) or re.search(VEHICLE_TYPES, t):
            self.vehicle = True
        if re.search(LOC_HINTS, t):
            self.location = True
        if re.search(ISSUE_HINTS, t):
            self.issue = True
        if re.search(r"(right now|asap|urgent|immediately|today|tonight|this (morning|evening)|blocking|in traffic)", t):
            self.urgency = True

    def need(self):
        if not self.vehicle: return "vehicle"
        if not self.location: return "location"
        if not self.issue: return "issue"
        if not self.urgency: return "urgency"
        return None

    def next_prompt(self):
        need = self.need()
        if need is None:
            return "Thanks. I can get a truck headed your way. What’s a good callback number in case we get disconnected?"
        options = {
            "vehicle": [
                "Thanks, I hear you. What is the vehicle year, make, and model?",
                "Got it. Can you tell me the vehicle year, make, and model?",
                "Okay. What are the year, make, and model of the vehicle?"
            ],
            "location": [
                "Where are you exactly—an address, intersection, or nearby business?",
                "What’s your exact location or the nearest cross-street?",
                "Tell me where the vehicle is—an address or a landmark works."
            ],
            "issue": [
                "What happened with the vehicle—flat tire, won’t start, accident, or something else?",
                "What seems to be the issue with the vehicle?",
                "Tell me what’s going on with the vehicle so we send the right truck."
            ],
            "urgency": [
                "Is this urgent right now, or is the vehicle in a safe spot?",
                "Do you need help immediately, or can it wait a bit?",
                "How urgent is it—are you blocking traffic or in a safe place?"
            ]
        }
        arr = options[need]
        p = arr[self.prompt_rephrase_ix % len(arr)]
        self.prompt_rephrase_ix += 1
        return p

# ---------- Outbound Speaker with soft-barge & ducking ----------

