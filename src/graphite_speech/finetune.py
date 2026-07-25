#!/usr/bin/env python
"""
Finetuning Granite Speech
=========================

Converted from a Jupyter notebook into a standalone, argument-driven script.

Original notebook by Avihu Dekel (https://huggingface.co/Avihu), with thanks to
Avishai Elmakies, George Saon, Alexander Brooks and Eliyahu Schwartz.

This script:
  1. Loads a subset of GigaSpeech (or another audio/text dataset with the same schema).
  2. Loads Granite Speech (model + processor).
  3. Optionally computes WER before finetuning.
  4. Finetunes the LoRA adapters + projector (or the full model, if requested).
  5. Optionally computes WER after finetuning and reports the improvement.

Example usage
-------------
    python finetune_granite_speech.py \\
        --model_name ibm-granite/granite-4.0-1b-speech \\
        --train_samples 5000 --val_samples 200 --test_samples 200 \\
        --max_steps 300 --learning_rate 3e-5 \\
        --output_dir save_dir

Set HF_TOKEN in the environment (or pass --hf_token) if the model/dataset requires
authentication. If running on Kaggle with a stored secret, pass --use_kaggle_secret.
"""

import argparse
import os
import json
import torch
import tqdm
import evaluate
from torch.utils.data import DataLoader
from datasets import Dataset, Audio
from whisper.normalizers import EnglishTextNormalizer
from transformers import TrainingArguments, Trainer
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.granite_speech import (
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechProcessor,
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}"


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Granite Speech on a GigaSpeech-style ASR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Auth -------------------------------------------------------------- #
    auth = parser.add_argument_group("Authentication")
    auth.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_API_KEY"),
        help="Hugging Face token. Defaults to $HF_API_KEY if set.",
    )
    auth.add_argument(
        "--use_kaggle_secret",
        action="store_true",
        help="If set, fetch the HF token from Kaggle's UserSecretsClient (secret name: HF_API_KEY) "
             "instead of --hf_token / env vars.",
    )
    auth.add_argument(
        "--kaggle_secret_name",
        type=str,
        default="HF_API_KEY",
        help="Name of the Kaggle secret holding the HF token (only used with --use_kaggle_secret).",
    )

    # --- Model --------------------------------------------------------------#
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--model_name", type=str, default="ibm-granite/granite-4.0-1b-speech",
        help="Model repo id or local path.",
    )
    model_grp.add_argument(
        "--full_finetune", action="store_true",
        help="If set, train all parameters instead of only LoRA + projector layers.",
    )

    # --- Dataset ------------------------------------------------------------#
    data_grp = parser.add_argument_group("Dataset")
    data_grp.add_argument("--train_manifest", type=str,
                           help="Train manifest file path")
    data_grp.add_argument("--val_manifest", type=str,
                           help="Validation manifest file path")
    data_grp.add_argument("--test_manifest", type=str,
                           help="Test manifest file path")
    data_grp.add_argument(
        "--instruction", type=str,
        default="Please transcribe the following audio to text<|audio|>",
        help="Instruction prompt prepended to each example.",
    )
    data_grp.add_argument("--preprocessing_num_workers", type=int, default=4)

    # --- Training hyperparameters -------------------------------------------#
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--output_dir", type=str, default="./save_dir")
    train_grp.add_argument("--per_device_train_batch_size", type=int, default=1)
    train_grp.add_argument("--per_device_eval_batch_size", type=int, default=1)
    train_grp.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_grp.add_argument("--num_train_epochs", type=float, default=1.0)
    train_grp.add_argument("--max_steps", type=int, default=300)
    train_grp.add_argument("--warmup_steps", type=int, default=50)
    train_grp.add_argument("--logging_steps", type=int, default=100)
    train_grp.add_argument("--eval_steps", type=int, default=100)
    train_grp.add_argument("--save_steps", type=int, default=100)
    train_grp.add_argument("--save_total_limit", type=int, default=2)
    train_grp.add_argument("--learning_rate", type=float, default=3e-5)
    train_grp.add_argument("--dataloader_num_workers", type=int, default=4)
    train_grp.add_argument("--data_seed", type=int, default=42)
    train_grp.add_argument("--bf16", action="store_true", default=True)
    train_grp.add_argument("--no_bf16", dest="bf16", action="store_false")

    # --- Evaluation / WER ----------------------------------------------------#
    eval_grp = parser.add_argument_group("Evaluation")
    eval_grp.add_argument("--wer_batch_size", type=int, default=16,
                           help="Batch size used for WER inference passes.")
    eval_grp.add_argument("--num_beams", type=int, default=4)
    eval_grp.add_argument("--max_new_tokens", type=int, default=400)
    eval_grp.add_argument("--skip_wer_before", action="store_true",
                           help="Skip computing WER before finetuning.")
    eval_grp.add_argument("--skip_wer_after", action="store_true",
                           help="Skip computing WER after finetuning.")
    eval_grp.add_argument("--skip_training", action="store_true",
                           help="Only run evaluation (e.g. to just measure baseline WER).")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def login_hf(args):
    from huggingface_hub import login

    token = args.hf_token
    if args.use_kaggle_secret:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret(args.kaggle_secret_name)

    if token:
        login(token=token)
    else:
        print("No HF token provided; proceeding without login "
              "(this will fail for gated models/datasets).")


# --------------------------------------------------------------------------- #
# Data preprocessing
# --------------------------------------------------------------------------- #
def process_gigaspeech_transcript(text):
    text = text.replace(" <COMMA>", ",")
    text = text.replace(" <PERIOD>", ".")
    text = text.replace(" <QUESTIONMARK>", "?")
    text = text.replace(" <EXCLAMATIONPOINT>", "!")
    text = text.lower()
    return text


def prep_example(example, tokenizer, instruction):
    chat = [dict(role="user", content=instruction)]
    example["prompt"] = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
    )
    example["text"] = process_gigaspeech_transcript(example["text"])
    return example


def prepare_dataset(ds, processor, instruction, num_workers):
    columns_to_remove = [col for col in ds.column_names if col not in ["audio", "text"]]
    ds = ds.cast_column("audio", Audio(sampling_rate=processor.audio_processor.sampling_rate))
    ds = ds.map(
        prep_example,
        fn_kwargs=dict(tokenizer=processor.tokenizer, instruction=instruction),
        remove_columns=columns_to_remove,
    )
    ds = ds.filter(lambda x: x["text"] not in ["<other>", "<noise>", "<music>", "<sil>"])
    return ds

def load_dataset(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    ds_items = [{"text": item["sentence"], "audio": item["path"]} for item in data]
    return Dataset.from_list(ds_items)

def load_and_prepare_data(args, processor):
    train_dataset = load_dataset(args.train_manifest)
    val_dataset = load_dataset(args.val_manifest)
    test_dataset = load_dataset(args.test_manifest)

    train_dataset = prepare_dataset(
        train_dataset, processor, args.instruction, args.preprocessing_num_workers
    )
    val_dataset = prepare_dataset(
        val_dataset, processor, args.instruction, args.preprocessing_num_workers
    )
    test_dataset = prepare_dataset(
        test_dataset, processor, args.instruction, args.preprocessing_num_workers
    )
    return train_dataset, val_dataset, test_dataset


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #
def extract_audio_array(audio):
    """Handle both legacy dict format and new torchcodec AudioDecoder objects."""
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        return samples.data.squeeze(0).numpy()
    elif isinstance(audio, dict):
        return audio["array"]
    return audio


class GraniteCollator:
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, examples):
        prompts = [example["prompt"] for example in examples]
        audios = [extract_audio_array(example["audio"]) for example in examples]

        processed = self.processor(
            prompts, audios, return_tensors="pt", padding=True, padding_side="left"
        )
        input_ids = processed.input_ids
        attention_mask = processed.attention_mask
        labels = None

        if not self.inference_mode:
            targets = [example["text"] + self.processor.tokenizer.eos_token for example in examples]
            targets = self.processor.tokenizer(targets, return_tensors="pt", padding=True, padding_side="right")
            input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
            attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
            labels = targets.input_ids.clone()
            labels[~(targets.attention_mask.bool())] = -100
            labels = torch.cat([torch.full_like(processed.input_ids, -100), labels], dim=1)

        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": processed.input_features,
            "input_features_mask": processed.input_features_mask,
        })


# --------------------------------------------------------------------------- #
# WER computation
# --------------------------------------------------------------------------- #
def compute_wer(model, processor, cur_dataset, batch_size, num_workers, num_beams, max_new_tokens):
    collator = GraniteCollator(processor, inference_mode=True)
    dataloader = DataLoader(cur_dataset, batch_size=batch_size, collate_fn=collator, num_workers=num_workers)
    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")
    model = model.eval().to(device)

    all_outputs = []
    for batch in tqdm.tqdm(dataloader, desc="Running inference"):
        batch = batch.to(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **batch, max_new_tokens=max_new_tokens, num_beams=num_beams, early_stopping=True
            )
        input_length = batch.input_ids.shape[1]
        outputs = outputs[:, input_length:].cpu()
        for x in outputs:
            all_outputs.append(processor.tokenizer.decode(x, skip_special_tokens=True))

    gt_texts = [normalizer(x) for x in cur_dataset["text"]]
    all_outputs = [normalizer(x) for x in all_outputs]
    wer = wer_metric.compute(references=gt_texts, predictions=all_outputs)
    return wer


# --------------------------------------------------------------------------- #
# Model loading / training
# --------------------------------------------------------------------------- #
def load_model_and_processor(args):
    processor = GraniteSpeechProcessor.from_pretrained(args.model_name)
    model = GraniteSpeechForConditionalGeneration.from_pretrained(args.model_name, dtype=torch.bfloat16)
    return model, processor


def set_trainable_params(model, full_finetune):
    for n, p in model.named_parameters():
        if full_finetune:
            p.requires_grad = True
        else:
            # only train the projector/lora layers
            p.requires_grad = "projector" in n or "lora" in n


def build_trainer(args, model, processor, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        report_to="none",
        bf16=args.bf16,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        data_seed=args.data_seed,
    )
    data_collator = GraniteCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    return trainer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    
    if local_rank == 0:
        for k,v in vars(args).items():
            print(f"{k}: {v}")
            
    login_hf(args)

    print(f"[rank {local_rank}] Loading model and processor: {args.model_name}")
    model, processor = load_model_and_processor(args)

    print(f"[rank {local_rank}] Loading and preparing dataset...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(args, processor)

    wer_before_train = None
    if not args.skip_wer_before and local_rank == 0:
        print("Computing WER before finetuning...")
        wer_before_train = compute_wer(
            model, processor, test_dataset,
            batch_size=args.wer_batch_size,
            num_workers=args.dataloader_num_workers,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"WER before finetuning: {wer_before_train:.3f}")
        torch.cuda.empty_cache()

    if not args.skip_training:
        print("Setting trainable parameters...")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # check actual module names for the LLM decoder
            task_type="CAUSAL_LM",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        set_trainable_params(model, args.full_finetune)

        print("Building trainer and starting training...")
        trainer = build_trainer(args, model, processor, train_dataset, val_dataset)
        trainer.train()

        torch.cuda.empty_cache()

    if not args.skip_wer_after and local_rank == 0:
        print("Computing WER after finetuning...")
        wer_after_train = compute_wer(
            model, processor, test_dataset,
            batch_size=args.wer_batch_size,
            num_workers=args.dataloader_num_workers,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"WER after finetuning: {wer_after_train:.3f}")
        if wer_before_train is not None:
            print(f"WER improvement: {(wer_before_train - wer_after_train):.3f}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()