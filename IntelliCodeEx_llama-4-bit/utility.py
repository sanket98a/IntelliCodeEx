from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline, LlamaCpp,CTransformers
import torch

## Check the cuda
if torch.cuda.is_available():
    device_type = "cuda:0"
else:
    device_type = "cpu"


# prompt special token

class CodeExplainer():
    def __init__(self) -> None:
        self.B_INST="[INST]"
        self.E_INST="[/INST]"
        self.B_SYS="<<SYS>>\n"
        self.E_SYS="\n<</SYS>>\n\n"
    def get_prompt(self,message: str, system_prompt: str,instruction_prompt:str) -> str:
        """
        Generate a comprehensive prompt for instructing a model to explain code.
        ------------------------------------------------------------------------------------
        Parameters:
        message (str): The user-provided code that requires an explanation.
        system_prompt (str): The system's prompt to set context for code explanation.
        ------------------------------------------------------------------------------------
        Returns:
        str: A formatted prompt for code explanation.
        --------------------------------------------------------------------------------------
        This method generates a prompt for instructing a language model to explain code. It combines
        the system's context prompt with the user-provided code and additional instructions.
        --------------------------------------------------------------------------------------
        Example usage:
        >>> user_code = "print('Hello, World!')"
        >>> system_context = "Explain the following Python code:"
        >>> prompt = get_prompt(user_code, system_context)
        >>> print(prompt)
        [INST] <<SYS>>
        Explain the following Python code:
        <</SYS>>

        user provided code to explain: print('Hello, World!')
        please explain the above code, following the instructions given by the user:
        [INSTRUCTION_PROMPT]
        [/INST]
        """
        texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        texts.append(f"user provided code to explain:{message.strip()}\n please explain the above code, following the instructions given by user:\n {instruction_prompt}[/INST]")
        return " ".join(texts)

    def calculate_tokens_per_second(self,total_time, total_tokens):
        """
        Calculate the rate of tokens processed per second.

        Parameters:
        total_time (float): The total time taken to process the tokens in seconds.
        total_tokens (int): The total number of tokens processed.

        Returns:
        float: The rate of tokens processed per second.

        This method computes the rate at which tokens are processed per second by dividing
        the total number of tokens processed by the total time taken in seconds.

        Example usage:
        >>> time_elapsed = 10.5  # Total time taken in seconds
        >>> tokens_processed = 4200  # Total number of tokens processed
        >>> rate = calculate_tokens_per_second(time_elapsed, tokens_processed)
        >>> print(f"Tokens processed per second: {rate} tokens/second")
        """
        tokens_per_second = total_tokens / total_time
        return tokens_per_second

    def llama_model(self, model_id: str = "TheBloke/Llama-2-7B-Chat-GGML",
                     model_basename: str = "llama-2-7b-chat.ggmlv3.q4_0.bin",
                       max_new_tokens: int = 2014, 
                       temperature: float = 0.7,
                       n_batch:int=500,
                      n_gpu_layers:int=60 ):
        """
        Loads the Llama GGML model using the LlamaCpp framework.

        Parameters:
            model_id (str): The model identifier for Hugging Face Hub. Default is "TheBloke/Llama-2-7B-Chat-GGML".
            model_basename (str): The base name of the model file. Default is "llama-2-7b-chat.ggmlv3.q4_0.bin".
            max_new_tokens (int): The maximum number of new tokens to generate. Default is 2014.
            temperature (float): The temperature for generating responses. Default is 0.7.
            n_batch (int): The batch size for the model when using CUDA:0. Default is 500.
            n_gpu_layers (int): The number of GPU layers to use, especially relevant for CUDA:0. Default is 60.

        Returns:
            LlamaCpp: An instance of the LlamaCpp class with the specified parameters.

        This method downloads the model from the Hugging Face Hub, sets various model parameters,
        and customizes these parameters based on the device type. It then loads the GGML model
        and returns it as an instance of the LlamaCpp class.
        """
        # Download the model from the Hugging Face Hub
        model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
        # Define model parameters
        kwargs = {
            "model_path": model_path,
            "n_ctx": 2014,
            "max_tokens": max_new_tokens
            }
        # Customize parameters based on the device type
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = n_gpu_layers

        if device_type.lower() == "cuda:0":
            kwargs["n_gpu_layers"] = n_gpu_layers
            kwargs["n_batch"] =n_batch
            kwargs["temperature"] = temperature

        # Print a success message
        print("GGML Model Loaded Successfully.")

        # Return the LlamaCpp instance with the specified parameters
        return LlamaCpp(**kwargs)




# find the input prompt token length
# def get_input_token_length(final_prompt: str) -> int:
#     input_ids = tokenizer([final_prompt], return_tensors='np', add_special_tokens=False)['input_ids']
#     return input_ids.shape[-1]

