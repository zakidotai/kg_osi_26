import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_MODEL = "Qwen/Qwen3.5-0.8B"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_BATCH_SIZE = 64
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE, model_kwargs={"dtype": DTYPE})
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
tokenizer.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=DTYPE).to(DEVICE).eval()

def _messages(text, question):
    prompt = (
        "You answer questions about a text using only that text. "
        'If the text does not contain the answer, answer exactly "NA" '
        "Keep each answer to one sentence or less."
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"TEXT:\n{text}\n\nQUESTION: {question}"},
    ]

def _clean_answer(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return raw.strip() or "NA"


def _gpu_seconds(texts, questions):
    return min(120, 20 + 2 * len(texts) * len(questions))

def answer_and_embed(texts, questions):
    prompts = [
        tokenizer.apply_chat_template(_messages(t, q), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        for t in texts
        for q in questions
    ]
    answers = []
    for i in range(0, len(prompts), GEN_BATCH_SIZE):
        batch = tokenizer(prompts[i : i + GEN_BATCH_SIZE], return_tensors="pt", padding=True)
        batch = batch.to(llm.device)
        with torch.inference_mode():
            out = llm.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, batch["input_ids"].shape[1] :]
        answers += [_clean_answer(a) for a in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]

    vecs = embedder.encode(answers, normalize_embeddings=True, batch_size=32)
    return answers, np.asarray(vecs, dtype=np.float32)


class QAEmbedder:
    def __init__(self,key):
        """key is the text key in the dataset"""
        self.key = key
        self.questions = []
        self.F = []

    def get_f(self,i,j,d):
        self.questions.append(q)
        return self.pop_a
    

    def embed(self):
        answers = []
        embeddings = []