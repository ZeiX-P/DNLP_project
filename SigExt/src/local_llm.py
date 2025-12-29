import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict

# -------------------------------------------------------------------------
# Registry of available LLM models
# Each entry maps a short model name to its Hugging Face model identifier
# and the style of instruction-following it uses.
# -------------------------------------------------------------------------
MODEL_REGISTRY = {
    "mistral": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "instruct_style": "mistral",
    },
    "qwen": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "instruct_style": "chatml",
    },
    "llama": {
        "hf_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "instruct_style": "llama",
    },
    #"deepseek": {
    #    "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #    "instruct_style": "chatml",
    #}, run a test and then discarded
}

ACTIVE_MODEL_NAME = None
LLM_MODEL = None
LLM_TOKENIZER = None
LLM_DEVICE = 'cpu'


def load_llm_model(model_name: str):
    global LLM_MODEL, LLM_TOKENIZER, LLM_DEVICE, ACTIVE_MODEL_NAME

    # If a model is already loaded, do nothing
    if LLM_MODEL is not None:
        return

    assert model_name in MODEL_REGISTRY, f"Unknown model: {model_name}"

    ACTIVE_MODEL_NAME = model_name
    model_cfg = MODEL_REGISTRY[model_name]
    hf_name = model_cfg["hf_name"]

    LLM_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(LLM_DEVICE)
    print(f"Attempting to load on this device: {LLM_DEVICE}")

    if LLM_DEVICE.type == 'cpu':
        print("WARNING: GPU non available. The model is run on CPU")
        dtype = torch.bfloat16
        quant_config = None
    else:
        # Use 4-bit quantization on GPU for memory efficiency
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        dtype = None

    try:
        print(f"Loading of {model_name}...")
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto"  # Automatically map layers to available GPU(s)
        )
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(hf_name)
        LLM_TOKENIZER.pad_token = LLM_TOKENIZER.eos_token

        print("Model loaded")

    except Exception as e:
        print(f"ERROR during loading of the model: {e}")
        LLM_MODEL = None
        LLM_TOKENIZER = None


def predict_one_local(x, model_name=None):

    # Ensure the model is loaded; if not, load it
    if LLM_MODEL is None:
        assert model_name is not None
        load_llm_model(model_name)

    prompt_text = x["prompt_input"]

    inputs = LLM_TOKENIZER(prompt_text, return_tensors="pt", add_special_tokens=False)

    model_inputs = inputs.to(LLM_MODEL.device)

    generated_ids = LLM_MODEL.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=LLM_TOKENIZER.eos_token_id,
    )

    decoded_output = LLM_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract only the response part, assuming instruction style uses [/INST] delimiter
    response_parts = decoded_output.split('[/INST]')
    if len(response_parts) > 1:
        response_text = response_parts[-1].strip()
    else:
        # Fallback if tag not found (rare for Mistral-Instruct)
        response_text = decoded_output.strip()

    return response_text