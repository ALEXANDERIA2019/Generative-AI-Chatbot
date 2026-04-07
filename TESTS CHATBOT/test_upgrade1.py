# User input 
#→ Intent classifier (LLM)
#     → FLOW_INTENT → motor de estados
#     → QA_INTENT → RAG
#     → SMALLTALK → respuesta generativa corta
# → validador de respuesta
# → humanizador
# → logging de métricas

import os
import json
import faiss
import numpy as np
import gradio as gr
from sklearn.preprocessing import normalize
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-ada-002"
TOP_K = 4

# =========================
# LOGGING
# =========================

LOG = []

def log_event(session, event, data):
    LOG.append({
        "time": str(datetime.utcnow()),
        "session": session,
        "event": event,
        "data": data
    })

# =========================
# RAG SETUP
# =========================

with open('json/library_data.json', encoding='utf-8') as f:
    data = json.load(f)

texts = [x["text"] for x in data]
meta = {i:x for i,x in enumerate(data)}

def embed(t):
    r = client.embeddings.create(model=EMBED_MODEL, input=t)
    return [d.embedding for d in r.data]

vecs = normalize(embed(texts))
vecs = np.array(vecs, dtype=np.float32)
index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)

def rag(query):
    q = normalize(embed([query]))
    q = np.array(q, dtype=np.float32)
    _, ids = index.search(q, TOP_K)
    out=[]
    for i in ids[0]:
        e=meta[i]
        out.append(e.get("answer", e.get("title","")+" "+e.get("status","")))
    return "\n".join(out)

# =========================
# INTENT CLASSIFIER
# =========================

def classify_intent(msg):
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role":"system","content":
             "Clasifica intención: FLOW, QA, SMALLTALK. Responde solo una palabra."},
            {"role":"user","content":msg}
        ]
    )
    return r.choices[0].message.content.strip()

# =========================
# HUMANIZADOR
# =========================

def humanize(text):
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.6,
        messages=[
            {"role":"system","content":
             "Redacta en español LATAM con tono humano, claro y cálido."},
            {"role":"user","content":text}
        ]
    )
    return r.choices[0].message.content.strip()

# =========================
# VALIDADOR SEMÁNTICO
# =========================

def validate(answer, context):
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role":"system","content":
             "¿La respuesta está respaldada por el contexto? responde SI o NO."},
            {"role":"system","content":context},
            {"role":"assistant","content":answer}
        ]
    )
    return "SI" in r.choices[0].message.content.upper()

# =========================
# FLOW ENGINE
# =========================

sessions = {}

FLOW_STATES = [
    "apertura","nombre","frecuencia","reaccion",
    "direccion","horario","pago","confirmacion","done"
]

def get_sess(sid):
    if sid not in sessions:
        sessions[sid]={"state":"apertura","vars":{},"steps":0}
    return sessions[sid]

def flow_step(msg, s):

    st=s["state"]; v=s["vars"]; s["steps"]+=1

    if st=="apertura":
        s["state"]="nombre"
        return humanize(
        "Gracias por confirmar tu interés. "
        "Haré unas preguntas breves. ¿Cuál es tu nombre completo?")

    if st=="nombre":
        v["nombre"]=msg
        s["state"]="frecuencia"
        return humanize(
        f"Gracias {msg}. ¿Cuántas veces al día tomas tu medicación?")

    if st=="frecuencia":
        v["freq"]=msg
        s["state"]="reaccion"
        if "una" in msg:
            base="Tratamiento de suministro mensual."
        elif "dos" in msg:
            base="Reposición mensual anticipada."
        else:
            base="Frecuencia se validará en consulta."
        return humanize(base+" ¿Has tenido reacción adversa reciente?")

    if st=="reaccion":
        v["reaccion"]=msg
        s["state"]="direccion"
        return humanize("¿Dirección de entrega?")

    if st=="direccion":
        v["dir"]=msg
        s["state"]="horario"
        return humanize("¿Horario de entrega preferido?")

    if st=="horario":
        v["horario"]=msg
        s["state"]="pago"
        return humanize("¿Forma de pago para el copago?")

    if st=="pago":
        v["pago"]=msg
        s["state"]="confirmacion"
        return humanize("¿Confirmamos inscripción al programa?")

    if st=="confirmacion":
        if msg.lower() in ["si","sí","confirmo"]:
            s["state"]="done"
            log_event("flow","completed",v)
            return humanize("Registro completado. Recibirás confirmaciones.")
        else:
            s["state"]="done"
            log_event("flow","cancelled",v)
            return humanize("Proceso cancelado. Puedes volver cuando desees.")

# =========================
# QA RAG
# =========================

def answer_rag(msg, history):
    ctx = rag(msg)

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        messages=[
            {"role":"system","content":
             "Responde solo con base en el contexto."},
            {"role":"system","content":ctx},
            *history,
            {"role":"user","content":msg}
        ]
    )

    ans = r.choices[0].message.content.strip()

    if not validate(ans, ctx):
        return "No tengo información suficiente para responder con certeza."

    return ans

# =========================
# SMALLTALK
# =========================

def smalltalk(msg):
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.7,
        messages=[
            {"role":"system","content":
             "Respuesta breve conversacional."},
            {"role":"user","content":msg}
        ]
    )
    return r.choices[0].message.content.strip()

# =========================
# ROUTER PRINCIPAL
# =========================

def chat(msg, history, sid):

    s = get_sess(sid)
    intent = classify_intent(msg)
    log_event(sid,"intent",intent)

    if s["state"]!="done" and intent=="FLOW":
        return flow_step(msg,s)

    if intent=="QA":
        return answer_rag(msg, history)

    return smalltalk(msg)

# =========================
# UI
# =========================

ui = gr.ChatInterface(
    fn=chat,
    additional_inputs=[gr.Textbox(value="session1")],
    chatbot=gr.Chatbot(type="messages"),
    title="Agente Conversacional Híbrido",
)

ui.launch(share=True)