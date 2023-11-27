# Import the required packages
from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
from langchain.chains import RetrievalQA
import logging
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import streamlit as st
import time
access_token_read ="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"
login(token = access_token_read)
logging.info("Login Successfully.")

if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"

########################## Set the streamlit page configuration #############################
st.set_page_config(page_title="Home", page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)


########################### Streamlit ui logo on side bar #####################################
with st.sidebar:
    st.markdown("""<div style='text-align: left; margin-top:-80px;margin-left:-40px;'>
    <img src="https://affine.ai/wp-content/uploads/2023/05/Affine-Logo.svg" alt="logo" width="300" height="60">
    </div>""", unsafe_allow_html=True)

############################# Tool Name ###############################################
st.markdown("""
    <div style='text-align: center; margin-top:-70px; margin-bottom: 5px;margin-left: -50px;'>
    <h2 style='font-size: 40px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    IntelliCodeEx
    </span>
    <span style='font-size: 40%;'>
    <sup style='position: relative; top: 5px; color: #ed4965;'>by Affine</sup>
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True)

################## Special Token of LLama 2 #################################
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" 

## Default Instruction Prompt for Code Explation ################################# 
Instruction_prompt = """Get the output in bullet points. Avoid the repetetion."""

########## Streamlit Sidebar ###############
with st.sidebar:
    # Select Programming language
    language = st.selectbox("Select Language :-",["C#","Javascript",".NET"])
    # model_name=st.selectbox("Select Model :-",["Llama-2-7B-Chat",'CodeLlama-7B-Instruct'])
    # Select Temperature
    temperature=st.slider("Temperature :-",0.0,1.0,0.7)
    #top_p=st.slider("top_p :-",0.0,1.0,0.95)
    #top_k=st.slider("top_k :- ",0,100,50)
    ## Final Instructions prompt
    INSTRUCTION_PROMPT=st.text_area("User Instruction :-",f"{Instruction_prompt}",height=100)

### Default system prompt
DEFAULT_SYSTEM_PROMPT=f"""You are {language} coding assistant. Assist the user by explaining.
if you don't know say, 'I don't Know the answer. please follow the user instruction."""

# create the custom prompt method
def get_prompt(
    message: str, system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # # for user_input, response in chat_history:
    # #     texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"## User provided code to explain:{message.strip()}\n ## Please explain the above code, following the instructions given by user:\n {INSTRUCTION_PROMPT} This important to return the results using markdown.[/INST]")
    # prompt=f"""[INST] please explain the user provided code in natural languge. Please wrap your code answer using ```. user provided code:{message}[/INST]"""

    return "".join(texts)


###################################### Self Quantization and Model Loading #######################################
# load_in_4bit=True,
# bnb_4bit_quant_type="nf4",
# bnb_4bit_compute_dtype=torch.bfloat16,
# bnb_4bit_use_double_quant=False

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
# Model ID
model_id = "meta-llama/Llama-2-7b-chat-hf"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model loading
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config,device_map={"":0})

# find the input prompt token length
def get_input_token_length(final_prompt: str) -> int:
    input_ids = tokenizer([final_prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]

# Loading the model
def load_llm():
    print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.15
    )
    hug_model=HuggingFacePipeline(pipeline=pipe)
    return hug_model

model=load_llm()


################################## Streamlit Ui #######################################
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            start=time.time()
            final_prompt=get_prompt(prompt,DEFAULT_SYSTEM_PROMPT)
            token_size=get_input_token_length(final_prompt)
            print("Token ::",token_size)
            print("Prompt :",final_prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            time_placeholder = st.empty()
            full_response = model.predict(final_prompt)
            end=time.time()
            total_time=end-start
            # for response in model.generate([final_prompt]):
            #     full_response += response
            #     message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            print(full_response)
            time_placeholder.markdown(f'**Time** :: {round(total_time,2)} Sec.')
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )





