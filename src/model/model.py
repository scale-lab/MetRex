
import os
import sys 
import tqdm
from typing import List

import openai 
import torch 
import numpy as np
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset 
from utils import parse_answer


def generate(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    prompts: List[str], 
    do_sample: bool = True, 
    temperature: float = 0.7, 
    max_new_tokens: int = 1024, 
    num_samples: int = 1
):
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"

    encoded_inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding='longest', 
        add_special_tokens=True
    )

    model_inputs = encoded_inputs.to('cuda')

    with torch.no_grad(): 
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_samples,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    tokenizer.padding_side = padding_side_default

    decoded_outputs = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    return decoded_outputs



def test_model(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    dataset: Dataset, 
    temperature: float, 
    num_samples: int = 1, 
    batch_size: int = 16
):
    model.eval()
    model.config.use_cache = True

    answers = []
    
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        prompts = batch['prompt']
        
        output = generate(
            model, 
            tokenizer, 
            prompts, 
            do_sample=True, 
            temperature=temperature, 
            max_new_tokens=1024, 
            num_samples=num_samples
        )
        answers.extend(output)
        
    return answers


class Model:
    
    model_instruct_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
    }
        

    def __init__(self, model_name: str, temperature: float = 0.4, num_samples: int = 10, load_model: bool =True) -> None:
        self.gpt_models = [
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0125',
            'gpt-4-turbo-2024-04-09',
            'gpt-4o-2024-05-13',
            'gpt-4-turbo'
        ]
    
        self.llama_models = [
            'llama3',
            'llama3.1',
            'llama3.1:70b',
            'llama3:8b'
        ]
        
        self.mistral_models = [
            'mistral',
            'mixtral'
        ]
        
        self.model_instruct_ids = {
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
            "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "llama3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
        }
        
        self.model_name = model_name
        self.temperature = temperature
        self.num_samples = num_samples 

        self.is_gpt = self.model_name in self.gpt_models
        if not self.is_gpt: 
            self.model_id = self.model_instruct_ids[self.model_name]
            if load_model: 
                self.model, self.tokenizer = self.load_model(
                    self.model_id
                )
            
    def message(self, prompt: str) -> List[str]: 
        if self.model_name in self.gpt_models: 
            batch_responses = []
            for pr in prompt: 
                response = self.call_gpt_model(
                    prompt=pr
                )
                batch_responses.append(response)
        else: 
            response = self.call_hf_model(
                prompt=prompt
            )
            batch_responses = []
            batch_size = len(prompt)
            for i in range(0, self.num_samples*batch_size, self.num_samples):
                response_parsed = [parse_answer(answer, self.model_name, self.llama_models, self.mistral_models) for answer in response[i:i+self.num_samples]]
                batch_responses.append(response_parsed)
                  
        return batch_responses 
    
    
    def call_gpt_model(self, prompt: str) -> List[str]:
        api_key = os.environ.get('OPENAI_KEY')
        client = openai.OpenAI(api_key=api_key)
        answers = []
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=self.temperature,
                n=self.num_samples,
                seed=0
            )
        
            for choice in response.choices:
                answers.append(choice.message.content)
            return answers
           
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}.", flush=True)
            sys.exit(0)


    def get_chat_template(self, system_prompt: str, user_prompt: str, assistant_answer: str = "") -> List[dict]:
    
        combined_prompt = user_prompt 
        if system_prompt: 
            combined_prompt = system_prompt + '\n\n' + user_prompt
            
        mistral_template = [{
            'role': 'user', 
            'content': combined_prompt
        }]

        llama_template = []
        if system_prompt: 
            llama_template = [{"role": "system", "content": system_prompt}]
        
        llama_template.append({"role": "user", "content": user_prompt})

        if assistant_answer: 
            mistral_template.append({"role": "assistant", "content": assistant_answer})
            llama_template.append({"role": "assistant", "content": assistant_answer})

        if self.model_name in self.llama_models:
            return llama_template 

        if self.model_name in self.mistral_models:
            return mistral_template 


    def apply_chat_template(self, messages: List[str], add_generation_prompt: bool = False):
        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        encodeds = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )
        return encodeds 
        
    def call_hf_model(self, prompt: str, max_tokens: int = 1024, top_k: int = 50, top_p: int = 1, temperature: float = 0.1, terminators=None, num_return_sequences: bool = 1):
        
        input_ids = self.tokenizer(
            prompt,
            padding='longest', 
            add_special_tokens=True, 
            return_tensors="pt"
        ).to(self.model.device)
       
        if terminators: 
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_tokens, 
                do_sample=True, 
                top_k=top_k,
                top_p=top_p,
                temperature=temperature, 
                num_return_sequences=self.num_samples
            )
        else:
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p, 
                temperature=temperature, 
                eos_tok_id=terminators,
                num_return_sequences=self.num_samples
            )

        self.tokenizer.padding_side = "right"

        answer = self.tokenizer.batch_decode(
            sequences=outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return answer
    
    def load_model(self, model_id, add_special_tokens=True): 
        access_token=os.environ.get('HF_ACCESS_TOKEN')
        cache_dir = os.environ.get('HF_CACHE_DIR')
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype, 
            bnb_4bit_use_double_quant=True, 
        ) 

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='cuda', 
            token=access_token, 
            quantization_config=bnb_config, 
            cache_dir=cache_dir
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=access_token, 
            cache_dir=cache_dir, 
        ) 
        if add_special_tokens: 
            tokenizer.pad_token = tokenizer.eos_token 
            tokenizer.padding_side = "right" 

        return model, tokenizer 
    