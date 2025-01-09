import os
from huggingface_hub import login
from datasets import load_dataset
import gc
import time
import torch
from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from qwen_vl_utils import process_vision_info
import wandb
from trl import SFTTrainer
from accelerate import Accelerator

# Initialize the accelerator
# accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="fp16")
# print(f"Distributed type: {accelerator.state.distributed_type}")
# print(accelerator.state)

def format_data(sample, system_message):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['question'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["answer"]
                }
            ],
        },
    ]

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    
    # for key in batch.keys():
    #     batch[key] = batch[key].to(device)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    # print({key: value.device for key, value in batch.items()})

    return batch  # Return the prepared batch

def sample_subset(dataset_split, n):
    return dataset_split.shuffle(seed=42).select(range(min(len(dataset_split), n)))



def main():
    os.environ['HF_HOME'] = '/iridisfs/scratch/sa5u24/LVLM'
    hf_home = os.path.expanduser(
        os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
    )
    print(hf_home)
    login(token="")

    system_message = """You are a Vision Language Model specialized in interpreting visual data from medical images. 
                Focus on delivering accurate, succinct, short answers based on the visual information. 
                Avoid additional explanation unless absolutely necessary."""

    # dataset_id = "HuggingFaceM4/ChartQA"
    # train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])

    # train_dataset = load_dataset("merve/vqav2-small",split="validation[:20%]")
    # eval_dataset = load_dataset("merve/vqav2-small",split="validation[30%:50%]")
    # test_dataset = load_dataset("merve/vqav2-small",split="validation[80%:]")

    dataset = load_dataset("flaviagiammarino/path-vqa")
    n_samples = 100

    # Sample 100 instances from each split
    train_subset = sample_subset(dataset["train"], n_samples)
    test_subset = sample_subset(dataset["test"], n_samples)
    validation_subset = sample_subset(dataset["validation"], n_samples)

    # Optional: Save the subsets or print them
    print("Sampled Train Dataset:", train_subset)
    print("Sampled Test Dataset:", test_subset)
    print("Sampled Validation Dataset:", validation_subset)

    train_dataset = [format_data(sample, system_message) for sample in train_subset]
    eval_dataset = [format_data(sample, system_message) for sample in validation_subset]
    test_dataset = [format_data(sample, system_message) for sample in test_subset]

    # without sampling
    # train_dataset = [format_data(sample, system_message) for sample in ds['train']]
    # eval_dataset = [format_data(sample, system_message) for sample in ds['validation']]
    # test_dataset = [format_data(sample, system_message) for sample in ds['test']]


    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    # model = model.to("cuda")
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    # Configure training arguments
    training_args = SFTConfig(
        output_dir="qwen2-7b-instruct-trl-sft-pvqa",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=3,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        #max_seq_length=1024  # Maximum sequence length for input
        # dataloader_pin_memory=False,  # Disable pin_memory here

        )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    wandb.init(
        project="qwen2-7b-instruct-trl-sft-pvqa",  # change this
        name="qwen2-7b-instruct-trl-sft-pvqa",  # change this
        config=training_args,
    )


    # model = accelerator.prepare(model)
    # train_dataset = accelerator.prepare(train_dataset)
    # eval_dataset = accelerator.prepare(eval_dataset)


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
        # accelerator=accelerator,
        
    )

    trainer.train()

if __name__ == "__main__": 
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()









