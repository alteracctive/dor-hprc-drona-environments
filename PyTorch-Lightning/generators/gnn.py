import os
import builtins
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
    _trainer_main_block,
)

def _drona_msg(msg, level="warning"):
    if hasattr(builtins, "drona_add_message"):
        builtins.drona_add_message(msg, level)
    else:
        print(f"[{level.upper()}] {msg}")

def _gen_gnn_script(
    exp_name, graph_dataset_type, graph_data_path, gnn_hidden_dim, gnn_num_layers,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    gnn_layer_type="gcn",
):
    """Returns (train_script, prefetch_script_or_None)."""
    use_builtin = graph_dataset_type in ("JODIE", "QM9")
    pyg_name = graph_dataset_type
    data_dir = "./data"

    if not use_builtin and not graph_data_path:
        _drona_msg("Custom graph dataset path is required.", "error")
        return None, None

    if use_builtin:
        prefetch_script = None
        if pyg_name == "QM9":
            dataset_setup = f'''
        from torch_geometric.datasets import QM9
        import os
        path = os.environ.get("QM9_PATH", DATA_DIR)
        self.train_ds = QM9(root=path)
        self.val_ds = self.train_ds
        self.num_features = self.train_ds.num_features
        self.num_classes = 19
        self.is_gnn_benchmark = True'''
        else:  # JODIE
            dataset_setup = f'''
        from torch_geometric.datasets import JODIE
        import os
        path = os.environ.get("JODIE_PATH", os.environ.get("ImageNet2012_PT_PATH", DATA_DIR))
        dataset = JODIE(root=path, name="wikipedia")
        self.data = dataset[0]
        self.num_features = dataset.num_features
        self.num_classes = 2
        self.is_gnn_benchmark = False'''
    else:
        dataset_setup = f'''
        data_path = "{_py_str(graph_data_path)}"
        self.data = torch.load(os.path.join(data_path, "data.pt"))
        self.num_features = self.data.num_node_features
        self.num_classes = int(self.data.y.max().item()) + 1
        self.is_gnn_benchmark = False'''
        prefetch_script = None

    if gnn_layer_type == "gat":
        conv_class = "GATConv"
        conv_args = "hidden_channels, heads=1"
        in_conv_args = "in_channels, hidden_channels, heads=1"
        out_conv_args = "hidden_channels, out_channels, heads=1"
    elif gnn_layer_type == "graphsage":
        conv_class = "SAGEConv"
        conv_args = "hidden_channels, hidden_channels"
        in_conv_args = "in_channels, hidden_channels"
        out_conv_args = "hidden_channels, out_channels"
    else: # gcn
        conv_class = "GCNConv"
        conv_args = "hidden_channels, hidden_channels"
        in_conv_args = "in_channels, hidden_channels"
        out_conv_args = "hidden_channels, out_channels"

    model_block = f'''class LitModel(L.LightningModule):
    """Multi-layer Graph Neural Network using {conv_class} layers for node classification."""

    def __init__(self, in_channels, hidden_channels={gnn_hidden_dim},
                 out_channels=10, num_layers={gnn_num_layers}, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        
        self.convs = nn.ModuleList()
        # Input Layer: maps input node features to hidden dimensions
        self.convs.append({conv_class}({in_conv_args}))
        # Hidden Layers: perform node message-passing steps
        for _ in range(max(0, num_layers - 2)):
            self.convs.append({conv_class}({conv_args}))
        # Output Layer: maps hidden state to classification classes
        self.convs.append({conv_class}({out_conv_args}))

    def forward(self, x, edge_index):
        """
        Forward pass.
        x: Node feature matrix (num_nodes, in_channels)
        edge_index: Adjacency list representing graph connectivity (2, num_edges)
        """
        for conv in self.convs[:-1]:
            out = conv(x, edge_index)
            if isinstance(out, tuple): out = out[0]
            if out.size(-1) != self.hparams.hidden_channels:
                out = out.view(-1, 1, self.hparams.hidden_channels).mean(dim=1)
            x = out.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            
        out = self.convs[-1](x, edge_index)
        if isinstance(out, tuple): out = out[0]
        if out.size(-1) != self.hparams.out_channels:
            out = out.view(-1, 1, self.hparams.out_channels).mean(dim=1)
        return out

    def _shared_step(self, batch, stage):
        out = self(batch.x, batch.edge_index)
        
        # Determine if we evaluate the loss on a node-level mask (semi-supervised split)
        mask = getattr(batch, f"{{stage}}_mask", None)
        if mask is not None:
            loss = F.cross_entropy(out[mask], batch.y[mask])
            acc = (out[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
        else:
            loss = F.cross_entropy(out, batch.y)
            acc = (out.argmax(dim=-1) == batch.y).float().mean()
            
        # Log loss and accuracy metrics to current active loggers
        batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        self.log(f"{{stage}}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{{stage}}_acc", acc, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        # Using Adam optimizer with a default weight decay parameter for regularization
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
'''

    template = _load_template("gnn")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("__DATA_DIR__", _py_str(data_dir))
    train_script = train_script.replace("__LOG_DIR__", _py_str(log_dir))
    train_script = train_script.replace("__EXPERIMENT_NAME__", _py_str(exp_name))
    train_script = train_script.replace("__LR__", str(lr))
    train_script = train_script.replace("__NUM_WORKERS__", str(nw))
    train_script = train_script.replace("__MAX_EPOCHS__", str(ep))
    train_script = train_script.replace("__ACCELERATOR__", _py_str(acc))
    train_script = train_script.replace("__DEVICES__", str(dev))
    train_script = train_script.replace("__PRECISION__", str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"')
    train_script = train_script.replace("__LOG_EVERY_N_STEPS__", str(log_n))
    train_script = train_script.replace("__SEED__", str(seed_val))
    train_script = train_script.replace("__GNN_HIDDEN_DIM__", str(gnn_hidden_dim))
    train_script = train_script.replace("__GNN_NUM_LAYERS__", str(gnn_num_layers))
    train_script = train_script.replace("        # __DATASET_SETUP__", dataset_setup)
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    return train_script, prefetch_script
