import os
from datasets import load_dataset
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer
from transformers import TrainerCallback
from huggingface_hub import login
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def format_data(sample, prompt):
    # print("temp", prompt.format(question=sample['question']))
    return {"messages": [
                {
                    "role": "question",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.format(question=sample['question']),
                        },{
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "answer",
                    "content": [{"type": "text", "text": sample["multiple_choice_answer"]}],
                },
            ],
        }


def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch



def main():
    os.environ['HF_HOME'] = '/data/user-data/sa25729/vision_language_task/LVLM'

    hf_home = os.path.expanduser(
        os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
    )
    
    # Replace 'your-hf-token-here' with your actual Hugging Face token
    login(token="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")

    print(hf_home)
     
    
    prompt= """Answer the question based on the provided ##Question## and image. ##Question##: {question}"""

    ds_train = load_dataset("merve/vqav2-small",split="validation[:1%]")
    ds_val = load_dataset("merve/vqav2-small",split="validation[99%:]")
    print("dataset", len(ds_train), len(ds_val))
    
    
    dataset_train = [format_data(sample, prompt) for sample in ds_train]
    dataset_val = [format_data(sample, prompt) for sample in ds_val]
    
    
    # Hugging Face model id
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir="fine-tuned-visionllama-vqa2", # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=500,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        # tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        # push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True}, # important for collator
        load_best_model_at_end=True,  # Load the best model after training
        evaluation_strategy='epoch'  
        )
        
    args.remove_unused_columns=False
    trainer = SFTTrainer(
        model=model,
        args=args,
        # train_dataset=dataset_train,
        eval_dataset= dataset_val,
        data_collator=collate_fn,
        dataset_text_field="", # needs dummy value
        peft_config=peft_config,
        tokenizer=processor.tokenizer
    )
    
    trainer.train()

if __name__ == "__main__":
    main()