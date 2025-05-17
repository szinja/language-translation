# English–isiXhosa Neural Machine Translation

This project implements a Transformer-based Neural Machine Translation (NMT) model to translate from English to isiXhosa using the OPUS SADiLaR bilingual dataset. The model is fine-tuned from `Helsinki-NLP/opus-mt-en-xh` using the Hugging Face Transformers library and trained in PyTorch.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py --en_file english.txt --xh_file xhosa.txt --epochs 3
```

### 3. Translate a Sentence

```bash
python inference.py
# Enter text when prompted
```

### Streamlit Inference App for Demo

A Streamlit interface is provided to demonstrate live English-to-isiXhosa translation. Run using:

```bash
streamlit run streamlit_app.py
```

Author: Sisipho Zinja
[WandB Dashboard](https://wandb.ai/szinja-university-of-rochester/huggingface)

Based on Hugging Face Transformers and Bilingual English-isiXhosa OPUS data by [SADiLaR](https://hdl.handle.net/20.500.12185/525).
