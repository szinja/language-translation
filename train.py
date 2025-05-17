import argparse
from datasets import load_metric, Dataset
from transformers import (
    MarianTokenizer, MarianMTModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, TrainerCallback
)
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import time

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
    model_inputs = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["xh"], truncation=True, padding="max_length", max_length=128)
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
            "bleu": state.log_history[-1].get("eval_bleu")  # if present
        })

def main(args):
    wandb.init(project="en-xh-translation")

    data = load_data(args.en_file, args.xh_file)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)

    train_ds = to_dataset(train)
    val_ds = to_dataset(val)

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name)

    tokenized_train = train_ds.map(lambda x: preprocess(x['translation'], tokenizer), batched=True)
    tokenized_val = val_ds.map(lambda x: preprocess(x['translation'], tokenizer), batched=True)


    # BLEU metric function
    bleu = load_metric("sacrebleu")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # We can't decode -100 labels so we replace them
        labels = np.where(labels != -100, tokenizer.pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate BLEU score
        bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        return {"bleu": bleu_result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500, 
        save_total_limit=2,
        predict_with_generate=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        report_to="wandb",
        logging_dir="./logs",
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
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
    parser.add_argument("--output_dir", type=str, default="./en-xh-model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    main(args)
