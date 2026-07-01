import os
import re
import json
import requests
import pandas as pd
import joblib

from dotenv import load_dotenv
from jsonschema import validate
from jsonschema.exceptions import ValidationError


# ==========================
# ENV
# ==========================

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

if not API_KEY:
    raise ValueError(
        "LLM_API_KEY environment variable not found."
    )


# ==========================
# LOAD MODEL
# ==========================

MODEL_FILE = "best_model.pkl"

model = joblib.load(MODEL_FILE)


# ==========================
# LOAD TRAINING DATA
# ==========================

df = pd.read_csv("cleaned_data.csv")

TARGET = "Survived"

X = df.drop(columns=[TARGET])


# ==========================
# OPENROUTER
# ==========================

URL = "https://openrouter.ai/api/v1/chat/completions"


# ==========================
# JSON SCHEMA
# ==========================

SCHEMA = {

"type":"object",

"required":[

"prediction_label",
"confidence_level",
"top_reason",
"second_reason",
"next_step"

],

"properties":{

"prediction_label":{"type":"string"},

"confidence_level":{"type":"string"},

"top_reason":{"type":"string"},

"second_reason":{"type":"string"},

"next_step":{"type":"string"}

}

}


# ==========================
# GUARDRAILS
# ==========================

def has_pii(text):

    email = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

    phone = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(
        re.search(email,text)
        or
        re.search(phone,text)
    )


# ==========================
# LLM CALL
# ==========================

def call_llm(
system_prompt,
user_prompt,
temperature=0.0,
max_tokens=512
):

    if has_pii(user_prompt):

        print(
            "Input blocked: PII detected."
        )

        return None


    headers = {

"Authorization":f"Bearer {API_KEY}",

"Content-Type":"application/json"

}


    payload = {

"model":"openai/gpt-4o-mini",

"messages":[

{
"role":"system",
"content":system_prompt
},

{
"role":"user",
"content":user_prompt
}

],

"temperature":temperature,

"max_tokens":max_tokens

}


    response = requests.post(
        URL,
        headers=headers,
        json=payload
    )


    if response.status_code != 200:

        print(
            "API ERROR:",
            response.status_code
        )

        print(response.text)

        return None


    return response.json()["choices"][0]["message"]["content"]


# ==========================
# PROMPT
# ==========================

SYSTEM_PROMPT = """
You are a prediction explanation system.

Output ONLY valid JSON.

Required fields:

prediction_label
confidence_level
top_reason
second_reason
next_step
"""


# ==========================
# EXPLAIN
# ==========================

def explain(
features,
pred,
prob,
temp=0
):


    prompt = f"""

Features:
{json.dumps(features)}

Prediction:
{pred}

Probability:
{round(prob,3)}

Return ONLY valid JSON.

"""


    raw = call_llm(
        SYSTEM_PROMPT,
        prompt,
        temperature=temp
    )


    if raw is None:

        return {

k:None

for k

in

SCHEMA["required"]

}


    try:

        obj = json.loads(
            raw.strip()
        )

    except json.JSONDecodeError:

        print(
            "Invalid JSON"
        )

        return {

k:None

for k

in

SCHEMA["required"]

}


    try:

        validate(
            obj,
            SCHEMA
        )

        return obj


    except ValidationError as e:

        print(
            "Schema error:",
            e
        )

        return {

k:None

for k

in

SCHEMA["required"]

}


# ==========================
# CREATE 3 TEST INPUTS
# ==========================

tests = []


for i in [0,5,10]:

    row = X.iloc[i].to_dict()

    tests.append(row)


# ==========================
# RUN
# ==========================

results=[]


for row in tests:


    sample = pd.DataFrame(
        [row]
    )


    pred = int(
        model.predict(sample)[0]
    )


    prob = float(
        model.predict_proba(sample)[0][1]
    )


    temp0 = explain(
        row,
        pred,
        prob,
        temp=0
    )


    temp07 = explain(
        row,
        pred,
        prob,
        temp=0.7
    )


    results.append({

"input":row,

"prediction":pred,

"probability":prob,

"temp0":temp0,

"temp07":temp07

})


# ==========================
# SAVE
# ==========================

with open(
"explanation_results.json",
"w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print("\nFinished")
print(
    json.dumps(
        results,
        indent=2
    )
)