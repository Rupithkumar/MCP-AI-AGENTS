import random
import pytest

LABELS = ["card_lost", "refund_status", "change_pin", "transfer_failed"]
WORDS = {
    "card_lost": ["lost", "stolen", "missing", "card", "wallet"],
    "refund_status": ["refund", "money", "back", "returned", "pending"],
    "change_pin": ["pin", "change", "reset", "code", "new"],
    "transfer_failed": ["transfer", "failed", "bank", "sent", "declined"],
}


def make_rows(n_per_label, seed):
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        for _ in range(n_per_label):
            rows.append({"text": " ".join(rng.choices(WORDS[label], k=5)), "label": label})
    rng.shuffle(rows)
    return rows


@pytest.fixture(scope="session")
def tiny_lm(tmp_path_factory):
    """A tiny random Qwen2-style model and a tokenizer trained locally (no downloads)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from src.prompts import SYSTEM

    corpus = [SYSTEM, "Classify the message. Labels:"] + LABELS + [w for ws in WORDS.values() for w in ws] + ["user", "assistant", "system", "Message", "Intent"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus * 20, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=512, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    import torch
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg), fast


@pytest.fixture()
def prepared(tmp_path):
    from src.data import prepare, load_prepared
    train, test = make_rows(60, 1), make_rows(15, 2)
    prepare(train, test, "synthetic", num_labels=4, seed=1, test_size=100, out=tmp_path)
    return tmp_path, load_prepared(tmp_path)
