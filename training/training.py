from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

MODELS = {   # name: (checkpoint, learning rate)
    "bert":       ("bert-base-cased", 3e-5),
    "roberta":    ("roberta-base", 3e-5),
    "jobbert":    ("jjzha/jobbert-base-cased", 3e-5),
    "distilbert": ("distilbert-base-cased", 5e-5),
    "electra":    ("google/electra-base-discriminator", 5e-5),
}
EPOCHS, BS, MAXLEN = 5, 16, 128

def load_tok(ckpt):
    kw = {"add_prefix_space": True} if "roberta" in ckpt else {}
    return AutoTokenizer.from_pretrained(ckpt, **kw)

def build(tok, words_list, labels_list):          # label only the FIRST subword of each word
    enc = tok(words_list, is_split_into_words=True, truncation=True, max_length=MAXLEN)
    items = []
    for i, labs in enumerate(labels_list):
        prev, row = None, []
        for w in enc.word_ids(i):
            row.append(-100 if (w is None or w == prev) else L2I[labs[w]])
            prev = w
        item = {k: enc[k][i] for k in enc.keys()}
        item["labels"] = row
        items.append(item)
    return items

def make_collate(tok):
    def collate(batch):
        labels = [b["labels"] for b in batch]
        out = tok.pad([{k: v for k, v in b.items() if k != "labels"} for b in batch], return_tensors="pt")
        L = out["input_ids"].shape[1]
        out["labels"] = torch.tensor([l + [-100] * (L - len(l)) for l in labels])
        return out
    return collate

@torch.no_grad()
def predict_probs(model, tok, words_list, bs=64):  # -> list of (n_words, 5) probability arrays
    model.eval(); out = []
    for i in range(0, len(words_list), bs):
        chunk = words_list[i:i + bs]
        enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float()
        probs = torch.softmax(logits, -1).cpu().numpy()
        for j, words in enumerate(chunk):
            wp = np.zeros((len(words), len(LABELS)), dtype=np.float32); wp[:, 0] = 1.0   # default "O"
            seen = set()
            for t, w in enumerate(enc.word_ids(j)):
                if w is not None and w not in seen:
                    wp[w] = probs[j, t]; seen.add(w)
            out.append(wp)
    return out

to_labels = lambda probs: [repair([LABELS[i] for i in p.argmax(-1)]) for p in probs]

def train_model(name, ckpt, lr, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tok = load_tok(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(
        ckpt, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I).to(dev)
    loader = torch.utils.data.DataLoader(build(tok, *data["train"]), batch_size=BS, shuffle=True,
                                         collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = EPOCHS * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        f1 = f1_score(data["dev"][1], to_labels(predict_probs(model, tok, data["dev"][0])))
        print(f"  {name} epoch {ep+1}: dev F1 = {f1:.4f}")
        if f1 > best_f1:                           # keep the best epoch by dev F1
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok
