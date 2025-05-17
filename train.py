import argparse
from datasets import load_metric, Dataset
from transformers import (
    MarianTokenizer, MarianMTModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
import wandb

def load_data(en_path, xh_path):
    with open(en_path, encoding="utf-8") as f:
        en_lines = [line.strip() for line in f.readlines()]
    with open(xh_path, encoding="utf-8") as f:
        xh_lines = [line.strip() for line in f.readlines()]
    return [(en, xh) for en, xh in zip(en_lines, xh_lines) if en and xh]

def to_dataset(pairs):
    return Dataset.from_dict({"translation": [{"en": e, "xh": x} for e, x in pairs]})

def preprocess(examples, tokenizer):
    model_inputs = tokenizer(examples["en"], truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["xh"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir="./logs",
        report_to="wandb",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
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
