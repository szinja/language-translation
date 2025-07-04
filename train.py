import argparse
from datasets import Dataset
from transformers import (
    MarianTokenizer, MarianMTModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, TrainerCallback, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import re
import time
import os

# Loading dataset
def load_data(en_path, xh_path):
    with open(en_path, encoding="utf-8") as f:
        en_lines = [line.strip() for line in f.readlines()]
    with open(xh_path, encoding="utf-8") as f:
        xh_lines = [line.strip() for line in f.readlines()]
    return [(en, xh) for en, xh in zip(en_lines, xh_lines) if en and xh]

def to_dataset(pairs):
    en, xh = zip(*pairs)    # flatten to "{'en': 'This is English.', 'xh': 'Lo sisiXhosa.'}"
    return Dataset.from_dict({"en": en, "xh": xh})

def preprocess(examples, tokenizer):
    # Tokenize the input (source) text
    model_inputs = tokenizer(
        examples["en"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Tokenize the target text using the modern `text_target` argument
    labels = tokenizer(
        text_target=examples["xh"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Custom Wandb ouput 
class CustomWandbLogger(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            elapsed = time.time() - self.start_time
            wandb.log({
                "elapsed_minutes": elapsed / 60,
                "global_step": state.global_step,
                "learning_rate": logs.get("learning_rate", None),
                "training_loss": logs.get("loss", None),
            })

    def on_epoch_end(self, args, state, control, **kwargs):
        wandb.log({
            "epoch": state.epoch,
            "bleu": state.log_history[-1]["eval_bleu"] if state.log_history and "eval_bleu" in state.log_history[-1] else None
        })

def main(args):
    wandb.init(
        project="en-xh-translation",
        name=f"run-lr-{args.learning_rate}-ep-{args.epochs}",
        group="longer-training", # {longer-training, custom-tokenizer, everything else is baseline}
        config=vars(args)
    )

    data = load_data(args.en_file, args.xh_file)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    train_ds = to_dataset(train)
    val_ds = to_dataset(val)

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    tokenized_train = train_ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    tokenized_val = val_ds.map(lambda x: preprocess(x, tokenizer), batched=True)

    # BLEU metric function
    def compute_metrics(eval_preds):
        import evaluate
        bleu = evaluate.load("sacrebleu")
        preds, labels = eval_preds

        # Decode predictions and normalize whitespace
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [re.sub(r"\s+", " ", pred).strip() for pred in decoded_preds]

        # Replace -100 in labels with pad_token_id
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [re.sub(r"\s+", " ", label).strip() for label in decoded_labels]

        # Format as required by sacrebleu
        references = [[label] for label in decoded_labels]

        # Compute BLEU
        bleu_result = bleu.compute(predictions=decoded_preds, references=references)
        return {"bleu": bleu_result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="wandb",
        logging_dir="./logs",
        logging_strategy="epoch",
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        predict_with_generate=True,
        fp16=args.fp16
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    os.environ["WANDB_DIR"] = "/content/wandb"
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Adding my custom logging 
    trainer.add_callback(CustomWandbLogger())
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_file", type=str, default="english.txt")
    parser.add_argument("--xh_file", type=str, default="xhosa.txt")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-xh")
    parser.add_argument("--output_dir", type=str, default="/content/en-xh-model")
    parser.add_argument("--run_name", type=str, default="en-xh-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--metric_for_best_model", type=str, default="bleu")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    args = parser.parse_args()

    main(args)
