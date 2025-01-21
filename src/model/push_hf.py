"""Push finetuned LoRA adapters to hugging face
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM

def main(): 

    # Load your base model and LoRA adapter
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Area adapter
    adapter_path = "/oscar/data/sreda/mabdelat/LLM4PPA-EXP5/llama_area_r128/checkpoint-2500"
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)

    adapter_repo = "manarabdelatty/Llama3-MetRex-Area-8b"
    model.push_to_hub(adapter_repo)

    # Delay adapter
    adapter_path = "/oscar/data/sreda/mabdelat/LLM4PPA-EXP5/llama8b_delay_sky_r128/checkpoint-2800/"
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)

    # Save the adapter
    adapter_repo = "manarabdelatty/Llama3-MetRex-Delay-8b"
    model.push_to_hub(adapter_repo)

    # Power adapter
    adapter_path = "/oscar/data/sreda/mabdelat/LLM4PPA-EXP5/llama_static_power_r128/checkpoint-2600/"
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)

    # Save the adapter
    adapter_repo = "manarabdelatty/Llama3-MetRex-Static-Power-8b"
    model.push_to_hub(adapter_repo)



if __name__ == "__main__":
    main()