import os
import builtins
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
)

def _drona_msg(msg, level="warning"):
    if hasattr(builtins, "drona_add_message"):
        builtins.drona_add_message(msg, level)
    else:
        print(f"[{level.upper()}] {msg}")

def _gen_nlp_script(
    exp_name, nlp_ds_type, nlp_blt, nlp_path, nlp_max_len,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    nlp_model_type="attention",
):
    """Returns (train_script, None)."""
    use_builtin = (nlp_ds_type == "builtin")
    data_dir = "./data"

    if not use_builtin and not nlp_path:
        _drona_msg("Custom NLP dataset path is required.", "error")
        return None, None

    # Dataset Setup
    if use_builtin:
        dataset_setup = f'''# Load cluster builtin text dataset
        import os
        import pandas as pd
        
        env_var = "fastText_PATH"
        path = os.environ.get(env_var, DATA_DIR)
        
        data_file = None
        if os.path.isdir(path):
            for r_dir, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith((".csv", ".tsv", ".txt")):
                        data_file = os.path.join(r_dir, f)
                        break
                if data_file:
                    break
        elif os.path.isfile(path):
            data_file = path

        texts, labels = [], []
        if data_file and os.path.exists(data_file):
            try:
                if data_file.endswith(".csv") or data_file.endswith(".tsv"):
                    sep = "\\t" if data_file.endswith(".tsv") else ","
                    df = pd.read_csv(data_file, sep=sep)
                    texts = df.iloc[:, 0].astype(str).tolist()
                    labels = df.iloc[:, 1].tolist()
                else:
                    with open(data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(None, 1)
                            if len(parts) == 2:
                                labels.append(parts[0])
                                texts.append(parts[1])
            except Exception as e:
                print(f"Error reading file {{data_file}}: {{e}}")

        if not texts:
            # Fallback to dummy data
            texts = ["this is a positive text example", "that was a negative response", "neutral sentence here"] * 50
            labels = [1, 0, 0] * 50

        if labels and isinstance(labels[0], str):
            unique_labels = sorted(list(set(labels)))
            label_map = {{lbl: i for i, lbl in enumerate(unique_labels)}}
            labels = [label_map[lbl] for lbl in labels]
            self.num_classes = len(unique_labels)
        else:
            self.num_classes = len(set(labels)) if labels else 2

        self.tokenizer.build_vocab(texts)
        split_idx = int(len(texts) * 0.8)
        self.train_ds = TextDataset(texts[:split_idx], labels[:split_idx], self.tokenizer, MAX_SEQ_LEN, is_seq2seq={nlp_model_type == "seq2seq"})
        self.val_ds = TextDataset(texts[split_idx:], labels[split_idx:], self.tokenizer, MAX_SEQ_LEN, is_seq2seq={nlp_model_type == "seq2seq"})'''
    else:
        dataset_setup = f'''# Load custom user text dataset
        import os
        import pandas as pd
        
        path = "{_py_str(nlp_path)}"
        data_file = path if os.path.isfile(path) else None
        if not data_file and os.path.isdir(path):
            for r_dir, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith((".csv", ".tsv", ".txt")):
                        data_file = os.path.join(r_dir, f)
                        break
                if data_file:
                    break

        texts, labels = [], []
        if data_file and os.path.exists(data_file):
            try:
                if data_file.endswith(".csv") or data_file.endswith(".tsv"):
                    sep = "\\t" if data_file.endswith(".tsv") else ","
                    df = pd.read_csv(data_file, sep=sep)
                    texts = df.iloc[:, 0].astype(str).tolist()
                    labels = df.iloc[:, 1].tolist()
                else:
                    with open(data_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(None, 1)
                            if len(parts) == 2:
                                labels.append(parts[0])
                                texts.append(parts[1])
            except Exception as e:
                print(f"Error reading custom file {{data_file}}: {{e}}")

        if not texts:
            # Fallback to dummy data
            texts = ["custom example text input here", "another custom sequence classification"] * 50
            labels = [1, 0] * 50

        if labels and isinstance(labels[0], str):
            unique_labels = sorted(list(set(labels)))
            label_map = {{lbl: i for i, lbl in enumerate(unique_labels)}}
            labels = [label_map[lbl] for lbl in labels]
            self.num_classes = len(unique_labels)
        else:
            self.num_classes = len(set(labels)) if labels else 2

        self.tokenizer.build_vocab(texts)
        split_idx = int(len(texts) * 0.8)
        self.train_ds = TextDataset(texts[:split_idx], labels[:split_idx], self.tokenizer, MAX_SEQ_LEN, is_seq2seq={nlp_model_type == "seq2seq"})
        self.val_ds = TextDataset(texts[split_idx:], labels[split_idx:], self.tokenizer, MAX_SEQ_LEN, is_seq2seq={nlp_model_type == "seq2seq"})'''

    # Model block depending on type
    if nlp_model_type == "seq2seq":
        model_block = f'''class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        seq_len = encoder_outputs.size(1)
        h_repeated = hidden.transpose(0, 1).repeat(1, seq_len, 1)
        attn_weights = F.softmax(self.attn(torch.cat((h_repeated, encoder_outputs), dim=-1)), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden

class LitModel(L.LightningModule):
    """Sequence-to-Sequence (Seq2Seq) model with dynamic attention routing."""
    def __init__(self, vocab_size, num_classes=2, embed_dim=128, hidden_dim=128, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = AttentionDecoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, src, tgt):
        max_len = tgt.size(1)
        encoder_outputs, hidden = self.encoder(src)
        outputs = []
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, max_len):
            logits, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            decoder_input = tgt[:, t].unsqueeze(1)
            
        return torch.cat(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].contiguous().view(-1), ignore_index=0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].contiguous().view(-1), ignore_index=0)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''
    else: # attention
        model_block = f'''class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(context)

class LitModel(L.LightningModule):
    """Text classification model with custom self-attention block."""
    def __init__(self, vocab_size, num_classes=2, embed_dim=128, num_heads=4, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = out + self.attn(out)
        out = self.norm(out)
        out = out.mean(dim=1)
        return self.fc(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)'''

    template = _load_template("nlp")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("__DATA_DIR__", _py_str(data_dir))
    train_script = train_script.replace("__LOG_DIR__", _py_str(log_dir))
    train_script = train_script.replace("__EXPERIMENT_NAME__", _py_str(exp_name))
    train_script = train_script.replace("__LR__", str(lr))
    train_script = train_script.replace("__BATCH_SIZE__", str(bs))
    train_script = train_script.replace("__NUM_WORKERS__", str(nw))
    train_script = train_script.replace("__MAX_EPOCHS__", str(ep))
    train_script = train_script.replace("__ACCELERATOR__", _py_str(acc))
    train_script = train_script.replace("__DEVICES__", str(dev))
    train_script = train_script.replace("__PRECISION__", str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"')
    train_script = train_script.replace("__LOG_EVERY_N_STEPS__", str(log_n))
    train_script = train_script.replace("__SEED__", str(seed_val))
    train_script = train_script.replace("__MAX_SEQ_LEN__", str(nlp_max_len))
    train_script = train_script.replace("__MODEL_TYPE__", _py_str(nlp_model_type))
    train_script = train_script.replace("        # __DATASET_SETUP__", dataset_setup)
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    return train_script, None
