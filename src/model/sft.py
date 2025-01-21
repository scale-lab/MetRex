"""Finetune LLMs on an instruction dataset using LoRA
"""
import os 
import yaml
import torch
import argparse 
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from accelerate import Accelerator
from unsloth import FastLanguageModel

from model.preprocess import preprocess
from model.model import Model, test_model
from model.eval import compute_acc_at_k_majority
from config.config import combine_cfgs
from plot import plot_loss 
from utils import get_target, set_seed
from huggingface_hub import login
from utils import get_logger, print_trainable_parameters

access_token = os.getenv('HF_ACCESS_TOKEN')
login(token=access_token)


def fine_tune_unsloth(
    model_name, 
    tokenizer, 
    dataset, 
    config, 
    output_dir,
    logger,
    checkpoint=None, 
):
   
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.train.max_seq_length,
        dtype=None,
        load_in_4bit=config.bnb.load_in_4bit,
    )
    
    print_trainable_parameters(model, logger)
    print(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.lora_r,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=config.train.gradient_checkpointing,
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    print_trainable_parameters(model, logger)
    print(model)
    
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    # call model before finetuning 
    # test_model(model, tokenizer, dataset['test'][0:20])
   
    model.config.use_cache = False

    if config.train.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        model.gradient_checkpointing_disable()
    
    training_arguments = TrainingArguments(
            per_device_train_batch_size=config.train.batch_size,
            per_device_eval_batch_size=config.train.batch_size,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
            num_train_epochs=config.train.epochs,
            warmup_steps=config.train.warmup_steps,
            dataloader_num_workers=config.train.dataloader_num_workers,
            max_steps=config.train.max_steps,
            learning_rate=config.train.learning_rate,
            lr_scheduler_type=config.train.lr_scheduler_type,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.train.logging_steps,
            eval_delay=0,
            logging_strategy=config.train.logging_strategy,
            eval_strategy=config.train.evaluation_strategy,
            save_strategy=config.train.save_strategy,
            save_steps=config.train.save_steps,
            eval_steps=config.train.eval_steps, 
            output_dir=output_dir,
            load_best_model_at_end=config.train.load_best_model_at_end,
            run_name=config.train.run_name,  # name of the W&B run (optional)
            report_to=config.train.report_to,
            gradient_checkpointing=config.train.gradient_checkpointing,  # Leads to reduction in memory at slighly decrease in speed
            # gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        dataset_text_field="prompt",
        args=training_arguments,
        packing = False,
        max_seq_length=config.train.max_seq_length
    )

    # Train model
    if checkpoint: 
        trainer.train(checkpoint)
    else:
        trainer.train()
    
    save_path = os.path.join(output_dir, f"{model_name}_finetuned")
    trainer.model.save_pretrained(save_path)
    
    plot_loss(trainer.state.log_history, os.path.join(output_dir, "loss.png"))

    # test the finetuned model on test set
    FastLanguageModel.for_inference(model)

    return model 




def fine_tune(
    model_name: str, 
    tokenizer: AutoTokenizer, 
    dataset, 
    config, 
    logger,
    checkpoint=None,
    output_dir="./results"
):
   
    # Load model in 4-bits and half precision
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = torch.bfloat16
    # getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.bnb.load_in_4bit, # Activates 4-bit precision loading
        bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type, # nf4
        bnb_4bit_compute_dtype=compute_dtype, # float16
        bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant, # False
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        quantization_config=bnb_config,
        device_map={"": Accelerator().process_index},
        # attn_implementation="flash_attention_2",
        # torch_dtype=torch.float16
    )
    model.config.pretraining_tp = 1

    print_trainable_parameters(model, logger)
    print(model)
    
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        r=config.lora.lora_r,
        use_dora=config.lora.use_dora,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model, logger)
    print(model)
    
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    # call model before finetuning 
    # test_model(model, tokenizer, dataset['test'][0:20])
   
    model.config.use_cache = False

    if config.train.gradient_checkpointing: 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    else:
        model.gradient_checkpointing_disable()
    
    training_arguments = TrainingArguments(
            per_device_train_batch_size=config.train.batch_size,
            per_device_eval_batch_size=config.train.batch_size,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps, # make finetuing 7 times slower !
            num_train_epochs=config.train.epochs,
            warmup_steps=config.train.warmup_steps,
            dataloader_num_workers=config.train.dataloader_num_workers,
            max_steps=config.train.max_steps,
            learning_rate=config.train.learning_rate,
            lr_scheduler_type=config.train.lr_scheduler_type,
            # fp16=not torch.cuda.is_bf16_supported(),
            # bf16=torch.cuda.is_bf16_supported(),
            # fp16=True,
            tf32=True,
            optim="paged_adamw_8bit",
            # fp16=not torch.cuda.is_bf16_supported(),
            # # Set whether to use 16-bit floating-point precision (fp16)
            # bf16=torch.cuda.is_bf16_supported(),
            # Set whether to use Bfloat16
            logging_steps=config.train.logging_steps,
            eval_delay=0,
            logging_strategy=config.train.logging_strategy,
            eval_strategy=config.train.evaluation_strategy,
            save_strategy=config.train.save_strategy,
            save_steps=config.train.save_steps,
            eval_steps=config.train.eval_steps, 
            output_dir=output_dir,
            load_best_model_at_end=config.train.load_best_model_at_end,
            run_name=config.train.run_name,  # name of the W&B run (optional)
            report_to=config.train.report_to,
            gradient_checkpointing=config.train.gradient_checkpointing,  # Leads to reduction in memory at slighly decrease in speed
            # gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        dataset_text_field="prompt",
        args=training_arguments,
        peft_config=peft_config,
        max_seq_length=config.train.max_seq_length
    )

    # Train model
    if checkpoint: 
        trainer.train(checkpoint)
    else:
        trainer.train()
        
    trainer.model.save_pretrained(os.path.join(output_dir, f"{model_name}_finetuned"))
    
    plot_loss(trainer.state.log_history, os.path.join(output_dir, "loss.png"))

    return model 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to config file for training", required=False, default="config/experiments/llama3_r64.yaml")
    parser.add_argument('--metric', type=str,  help="metric", default='area')
    parser.add_argument('--checkpoint', type=str, help="Resume finetuning from checkpoint", required=False, default=None)
    parser.add_argument('--use_sloth', action='store_const', const=True, help="Use sloth models for faster finetuning", default=False)
    parser.add_argument('--run_name', type=str, help="Name of the run", default=None)
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/sft")

    args = parser.parse_args()

    config_file = args.config 
    metric = args.metric
    checkpoint = args.checkpoint
    use_sloth = args.use_sloth
    run_name = args.run_name
    output_dir = args.output_dir
    
    config = combine_cfgs(config_file)

    config.DATASET.METRIC = metric 

    if run_name:
        config.train.run_name = run_name
    
    output_dir = os.path.join(output_dir, config.train.run_name)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(config.seed, n_gpu=torch.cuda.device_count())

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    shuffle = config.DATASET.SHUFFLE 
    model_name = config.LLM.NAME 

    model = Model(model_name)
    model_id = model.model_instruct_ids[model_name]
    
    logfile = os.path.join(output_dir, f"{model_name}")
    logger = get_logger(output=logfile)
    
    logger.info(f"BLUE: Number of GPUs {torch.cuda.device_count()}")
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=access_token, 
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token # end-of-sequence token
    tokenizer.padding_side = config.tokenizer.padding_side # pad to have max seq length
    
    # load dataset
    dataset = preprocess(
        config=config,
        tokenizer=tokenizer, 
        model=model, 
        filter_max_seq=True, 
        shuffle=shuffle
    )

    effective_batch_size = config.train.batch_size*config.train.gradient_accumulation_steps
    logger.info(f"BLUE: Training data length {len(dataset['train'])} Validation data length {len(dataset['valid'])} Testing data length {len(dataset['test'])}" )
    logger.info(f"BLUE: Training steps {(len(dataset['train'])/effective_batch_size)*config.train.epochs} ")

    if checkpoint:
        logger.info(f"Resuming finetuning from checkpoint {checkpoint}")
    
    # Run finetuning
    if use_sloth:
        model = fine_tune_unsloth(
            model_name=model_id, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            config=config, 
            logger=logger,
            checkpoint=checkpoint, 
            output_dir=output_dir
        )
    else:
        model = fine_tune(
            model_name=model_id, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            config=config, 
            logger=logger,
            checkpoint=checkpoint, 
            output_dir=output_dir
        )
    
    # test finetuned model 
    answers_bylevel = {}
    metric = config.DATASET.METRIC
    for level in dataset['test'].keys():
        answers = test_model(
            model, 
            tokenizer, 
            dataset['test'][level], 
            metric=metric, 
            logger=logger,
            temperature=config.LLM.TEMP, 
            batch_size=config.test.batch_size, 
            num_samples=config.test.num_samples
        )
        answers_bylevel[level] = answers

        targets = [get_target({metric: sample['target']}, metrics=[metric]) for sample in dataset['test'][level]]

        # reshape answers to list of lists
        answers_2d = [answers[i * config.test.num_samples:(i + 1) * config.test.num_samples] for i in range(config.test.batch_size)]
        
        for t in [0.05, 0.1, 0.2]: 
            acc_at_1, acc_at_5, acc_at_10 = compute_acc_at_k_majority(
                answers={metric: answers_2d}, 
                ground_truth=targets, 
                k=[1, 5, 10],
                tolerance=t,
                function=np.median
            )

            logger.info(f"WHITE: acc@1 t={t} {acc_at_1}")
            logger.info(f"WHITE: acc@5 t={t}: {acc_at_5}")
            logger.info(f"WHITE: acc@10 t={t}: {acc_at_10}")

    
if __name__ == "__main__":
    main()

