import gradio as gr
import requests as r
import json
from requests.structures import CaseInsensitiveDict

# Creation a chat frontend using Gradio.app
def call_api_single(text):

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"
    url = "http://127.0.0.1:8000/query" # Set here the URL of the API

    response = r.post(url,
                      headers = headers,
                      data = json.dumps({"text":text}))
    
    candidate = response.json()
    
    
    return candidate['passage'][0]

def chat(message, history,use_past = False):
    history = history or []
    new_message = ' '
    if use_past:
        for mess, res in history:
            new_message += mess + ' ' + res
        response = call_api_single(new_message + ' ' + message)
    else:
        response = call_api_single(message)
    history.append((message, response))
    return history, history


def gradio_chat():
    """
    Use Gradio to create a chatbot.
    """
    chatbot = gr.Chatbot(label = 'Chatbot')#.style(color_map=("green", "pink"))
    demo = gr.Interface(
        chat,
        ["text", "state"],
        [chatbot, "state"],
        allow_flagging="never",
    )

    demo.launch()

# Creation a candidate scoring frontend using Gradio.app
def call_api_multiple(text):

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"
    url = "http://127.0.0.1:8000/query" # Set here the URL of the API

    response = r.post(url,
                      headers = headers,
                      data = json.dumps({"text":text}))
    
    candidate = response.json()
    
    
    return dict(zip(candidate['passage'], candidate['scores']))
    
def gradio_list():

    examples = [["Who is the best founder ?"],["How to raise cash ?"],["When is product-market fit found ?"]]
                

    demo = gr.Interface(fn=call_api_multiple,
                        inputs=[gr.Textbox(lines=2, placeholder="Write your text here...")],
                        examples=examples,
                        description="A document classification interface powered by ML.",
                        title="Document Classification",
                        outputs=gr.Label())

    demo.launch()

