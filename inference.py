from transformers import MarianTokenizer, MarianMTModel

def translate(text, model_name="en-xh-model"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        text = input("Enter English sentence to translate (or type 'exit'): ")
        if text.strip().lower() == "exit":
            break
        print("Translation:", translate(text))
