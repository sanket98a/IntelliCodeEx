# Import the required packages
import streamlit as st
from streamlit_chat import message
from langchain.callbacks import StreamlitCallbackHandler
from utility import CodeExplainer
import time
# st_callback = StreamlitCallbackHandler(st.container())
# from langchain.callbacks.base import BaseCallbackHandler
# from streamlit.components.v1 import html
# import streamlit.components.v1 as components
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

########################## Set the streamlit page configuration #############################
st.set_page_config(page_title="Home", page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

########################### Streamlit ui logo on side bar #####################################
with st.sidebar:
    st.markdown("""<div style='text-align: left; margin-top:-80px;margin-left:-20px;'>
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
####################################################################################

## History
history=[]

####################################################################################
# Default set user instruction.
Instruction_prompt = """Get the output in bullet points. Avoid the repetetion."""

#################################### Sidebar ########################################
# set sidebar
with st.sidebar:
    # select the programing language
    language = st.selectbox("Select Language :-",["C#","Javascript",".NET","Python"])
    # Select the LLM model
    # model_name=st.selectbox("Select Model :-",["Llama-2-7B-Chat",'CodeLlama-7B-Instruct'])
    ## Set temperature settings. For Deterministic results set temperature close to 0. More dynamic results set close to 1.
    temperature=st.slider("Temperature :-",0.0,1.0,0.7)
    # set the Top-p parameter
    #top_p=st.slider("top_p :-",0.0,1.0,0.95)
    # set the parameter Top-k
    #top_k=st.slider("top_k :- ",0,100,50)
    ## User instruction box on streamlit ui
    INSTRUCTION_PROMPT=st.text_area("User Instruction :-",f"{Instruction_prompt}",height=200)

# Default Sys Prompt
# DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

## Set default system prompt
DEFAULT_SYSTEM_PROMPT=f"""You are {language} coding assistant. Assist the user by explaining.
if you don't know say, 'I don't Know the answer."""

# # Load the selected model
# if model_name=="Llama-2-7B-Chat":
#     print("Llama 7B model Loading")
#     model_id="TheBloke/Llama-2-7B-Chat-GGML"
#     model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"
# else:
#     print("CodeLlama-7B-Instruct-GGML model Loading")
#     model_id="TheBloke/CodeLlama-7B-Instruct-GGML"
#     model_basename="codellama-7b-instruct.ggmlv3.Q2_K.bin"

##################################### Model Loading ##################################################
## Code Explainer Object
CX=CodeExplainer()

## Llama-2 7B Chat GGML model for code explantation 
model_id="TheBloke/Llama-2-7B-Chat-GGML"
model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin"
print("Model Loading start")
model=CX.llama_model(model_id=model_id,model_basename=model_basename,temperature=temperature)
print("Load Model Successfully.")

##################################### ChatBot UI ###################################################
## ChatBot System and User History
if "messages" not in st.session_state:
    st.session_state.messages = []

## Streaming on UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## User Input
if prompt := st.chat_input("What is up?"):
        # Append the user input to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            start=time.time()
            # Design the final prompt
            final_prompt=CX.get_prompt(prompt,DEFAULT_SYSTEM_PROMPT,INSTRUCTION_PROMPT)
            # print(final_prompt)
        # Assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = model.predict(final_prompt)
            end=time.time()
            total_time=end-start
            # for response in model.generate([final_prompt]):
            #     full_response += response
            #     message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            # print the Full Response and Resonse time.
            print("*"*50)
            print(full_response)
            print(f"Total Time :: {round(total_time)} sec")
        # Update the assistant message into session state    
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )