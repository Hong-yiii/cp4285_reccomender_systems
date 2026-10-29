"""A frozen causal language model with LoRA adapters that recommends the next category.

A query is a user's history of category IDs. The prompt lists the category names, oldest
first, and every candidate category name, closed by a newline so the set is prefix-free,
is scored by its exact log-probability. One forward pass per user packs the prompt and all
candidates: a 4D attention mask lets each candidate attend only to the prompt and to its
own earlier tokens, and every candidate's positions continue from the end of the prompt.
Ranking is therefore exact over the closed category set, with no beam search or sampling.

Training is next-token cross-entropy on the target name's tokens. LoRA adapters hold every
trainable parameter, with dropout off, so updates are deterministic given the batches.
`add_stream` freezes the active adapters and stacks a new one (adapter isolation);
`set_active` chooses which adapters score, which measures rolling one back, and `drop`
deletes one so that later streams build without it.
"""

from collections.abc import Sequence
from copy import deepcopy

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from ..experiment.model import Recommender, candidate_indices

IMPLEMENTATION = "category-lm-lora-v1"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
HEADER = "Purchases by one shopper, oldest first:\n"
QUESTION = "Next purchase category:\n"


class CategoryLMLearner(Recommender[list[dict], list[list[int]], torch.Tensor]):
    """Own the adapters and AdamW; the base model's weights never change.

    Batches are lists of {"history": [category IDs], "target": category ID}. Category IDs
    are 0-based columns into `names`. One train call is one optimizer step.
    """

    def __init__(self, model, tokenizer, names, *, learning_rate, continuation_learning_rate,
                 rank=8, history=20, score_batch=8, device="cpu", model_id="", revision=""):  # fmt: skip
        self.names = list(names)
        self.tokenizer = tokenizer
        self.history = history
        self.score_batch = score_batch
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.continuation_learning_rate = continuation_learning_rate
        self.model_id, self.revision = model_id, revision
        self.config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                 target_modules=list(TARGET_MODULES), bias="none")  # fmt: skip
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = get_peft_model(model, deepcopy(self.config), adapter_name="a0").to(self.device)
        self.adapters, self.active = ["a0"], ["a0"]
        self.name_ids = [self.encode(name + "\n") for name in self.names]
        self.phase = "initial"
        self.steps = 0
        self.optimizer = self.new_optimizer(learning_rate)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, history):
        lines = "".join(f"- {self.names[c]}\n" for c in history[-self.history :])
        return [self.tokenizer.bos_token_id, *self.encode(HEADER + lines + QUESTION)]

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def new_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.trainable(), lr=learning_rate, weight_decay=0.0)

    def autocast(self):
        enabled = self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def train(self, batch: list[dict]) -> float:
        if not batch:
            raise ValueError("A training batch needs at least one example")
        rows = []
        for row in batch:
            prompt, target = self.prompt(row["history"]), self.name_ids[row["target"]]
            rows.append((prompt + target, [-100] * len(prompt) + target))
        width = max(len(ids) for ids, _ in rows)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.tensor([i + [pad] * (width - len(i)) for i, _ in rows], device=self.device)
        labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in rows], device=self.device)
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i, _ in rows],
                            device=self.device)  # fmt: skip
        self.model.train()
        with self.autocast():
            loss = self.model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        return float(loss.detach())

    def begin_continuation(self) -> None:
        if self.phase != "initial":
            raise RuntimeError("begin_continuation was already called")
        self.phase = "continuation"
        self.optimizer = self.new_optimizer(self.continuation_learning_rate)  # one AdamW reset

    def add_stream(self, name):
        """Freeze every adapter so far and stack a new trainable one on top."""
        if self.phase != "continuation":
            raise RuntimeError("Streams are added during continuation")
        if name in self.adapters:
            raise ValueError(f"Adapter {name} exists")
        self.model.add_adapter(name, deepcopy(self.config))
        self.adapters.append(name)
        self.set_active(self.adapters, trainable=name)
        self.optimizer = self.new_optimizer(self.continuation_learning_rate)

    def set_active(self, names, trainable=None):
        """Score and train with these adapters summed; only `trainable` (or the newest) learns."""
        names = list(names)
        if not names or any(n not in self.adapters for n in names):
            raise ValueError("Active adapters must be a nonempty subset of the learner's")
        self.model.base_model.set_adapter(names)
        trainable = trainable or self.active_trainable()
        for key, parameter in self.model.named_parameters():
            if "lora_" in key:
                parameter.requires_grad_(f".{trainable}." in key)
        self.active = names

    def drop(self, name):
        """Delete one stacked adapter for good (immediate rollback); later streams build without it."""
        if name == "a0" or name not in self.adapters:
            raise ValueError("Only a stacked stream adapter can be dropped")
        self.model.delete_adapter(name)
        self.adapters.remove(name)
        self.set_active(self.adapters)
        self.optimizer = self.new_optimizer(self.continuation_learning_rate)

    def active_trainable(self):
        return self.adapters[-1]

    def packed(self, history, columns):
        """Token IDs, positions, candidate owner per token, and each candidate's token span."""
        prompt = self.prompt(history)
        ids, positions, owner = list(prompt), list(range(len(prompt))), [-1] * len(prompt)
        spans = []
        for k, column in enumerate(columns):
            name = self.name_ids[column]
            spans.append((len(ids), len(name)))
            ids += name
            positions += range(len(prompt), len(prompt) + len(name))
            owner += [k] * len(name)
        return ids, positions, owner, len(prompt), spans

    @torch.no_grad()
    def score(self, queries: list[list[int]], /, candidates: Sequence[int] | None = None):
        columns = candidate_indices(candidates, len(self.names)) or tuple(range(len(self.names)))
        self.model.eval()
        lm = self.model.get_base_model()  # LoRA layers sit inside it, so adapters apply
        decoder, head = lm.get_decoder(), lm.get_output_embeddings()
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        out = []
        for start in range(0, len(queries), self.score_batch):
            packs = [self.packed(h, columns) for h in queries[start : start + self.score_batch]]
            width = max(len(p[0]) for p in packs)
            ids = torch.zeros((len(packs), width), dtype=torch.long)
            positions = torch.zeros_like(ids)
            allowed = torch.zeros((len(packs), 1, width, width), dtype=torch.bool)
            for b, (tokens, pos, owner, _, _) in enumerate(packs):
                n = len(tokens)
                ids[b, :n], positions[b, :n] = torch.tensor(tokens), torch.tensor(pos)
                own = torch.tensor(owner)
                same = (own[None, :] == -1) | (own[None, :] == own[:, None])
                allowed[b, 0, :n, :n] = torch.ones((n, n), dtype=torch.bool).tril() & same
                allowed[b, 0, n:, 0] = True  # padding rows attend to the first token only
            mask = torch.zeros(allowed.shape, dtype=dtype)
            mask.masked_fill_(~allowed, torch.finfo(dtype).min)
            with self.autocast():
                hidden = decoder(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                 position_ids=positions.to(self.device)).last_hidden_state  # fmt: skip
            for b, (tokens, _, _, prompt_length, spans) in enumerate(packs):
                sources, targets, owners = [], [], []
                for k, (start_at, length) in enumerate(spans):
                    # A candidate's first token is predicted at the prompt's last position.
                    sources += [prompt_length - 1, *range(start_at, start_at + length - 1)]
                    targets += tokens[start_at : start_at + length]
                    owners += [k] * length
                logp = torch.empty(len(sources))
                for c in range(0, len(sources), 1024):  # the head is vocabulary-wide; chunk it
                    rows = hidden[b, torch.tensor(sources[c : c + 1024], device=self.device)]
                    with self.autocast():
                        logits = head(rows)
                    picked = torch.tensor(targets[c : c + 1024], device=self.device)
                    chunk = logits.float().log_softmax(-1).gather(1, picked[:, None])[:, 0]
                    logp[c : c + len(chunk)] = chunk.cpu()
                row = torch.zeros(len(spans)).index_add_(0, torch.tensor(owners), logp)
                out.append(row)
        return torch.stack(out)

    def state(self):
        """Adapter weights, active set, phase and optimizer: everything an update changes."""
        weights = {n: {k: v.detach().cpu().clone() for k, v in
                       get_peft_model_state_dict(self.model, adapter_name=n).items()}
                   for n in self.adapters}  # fmt: skip
        return {"weights": weights, "adapters": list(self.adapters), "active": list(self.active),
                "phase": self.phase, "steps": self.steps,
                "optimizer": deepcopy(self.optimizer.state_dict())}  # fmt: skip

    def load_state(self, state):
        """Restore a state() of a learner over the same base model and categories."""
        for name in state["adapters"]:
            if name not in self.adapters:
                self.model.add_adapter(name, deepcopy(self.config))
                self.adapters.append(name)
        if self.adapters != state["adapters"]:
            raise ValueError("Adapter stacks differ; load into a learner built the same way")
        for name, weights in state["weights"].items():
            set_peft_model_state_dict(self.model, weights, adapter_name=name)
        self.phase, self.steps = state["phase"], state["steps"]
        self.set_active(state["active"])
        rate = self.learning_rate if self.phase == "initial" else self.continuation_learning_rate
        self.optimizer = self.new_optimizer(rate)
        self.optimizer.load_state_dict(state["optimizer"])

    def fork(self):
        """Copy the whole learner; for small test models. Large runs use state()/load_state()."""
        return deepcopy(self)

    def describe(self) -> dict:
        trainable = sum(p.numel() for p in self.trainable())
        return {
            "model": "category language model with LoRA adapters",
            "implementation": IMPLEMENTATION,
            "base_model": self.model_id, "base_revision": self.revision,
            "lora": {"rank": self.config.r, "alpha": self.config.lora_alpha, "dropout": 0.0,
                     "target_modules": list(TARGET_MODULES)},
            "adapters": list(self.adapters), "active": list(self.active),
            "trainable_parameters": trainable, "categories": len(self.names),
            "history": self.history, "phase": self.phase, "steps": self.steps,
            "update_unit": "one AdamW step on one batch of next-category examples",
            "scoring": "exact log-probability of every category name, packed in one pass per user",
            "state_policy": "base weights frozen; AdamW reset once at continuation and per new adapter",
        }  # fmt: skip
