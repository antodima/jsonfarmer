import lightning as L
import litgpt
import torch
from litgpt.lora import GPT
from torch import nn
from torchmetrics import Accuracy


class Jsonfarmer(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_tokens: int,
        lr: float = 5e-3,
        use_lora: bool = False,
    ):
        super().__init__()
        lora_params = {}
        if use_lora:
            lora_params = {
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "lora_query": True,
                "lora_key": False,
                "lora_value": True,
                "lora_projection": False,
                "lora_mlp": False,
                "lora_head": False,
            }

        self.lr = lr
        self.num_tokens = num_tokens
        self.model_name = model_name
        self.model = GPT.from_name(
            name=model_name,
            **lora_params,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(
            task="multiclass",
            num_classes=self.num_tokens,
            ignore_index=-100,
        )
        if use_lora:
            litgpt.lora.mark_only_lora_as_trainable(self.model)

    def on_train_start(self):
        state_dict = torch.load(
            f"checkpoints/{self.model_name}/lit_model.pth", mmap=True
        )
        self.model.load_state_dict(state_dict, strict=False)

    def loop_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)

        targets = targets[..., 1:].reshape(-1)
        logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
        loss = self.loss_fn(logits, targets)
        acc = self.acc_fn(
            torch.argmax(logits, dim=-1),
            targets,
        )

        return loss, acc

    def training_step(self, batch):
        loss, acc = self.loop_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch):
        loss, acc = self.loop_step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def configure_optimizers(self):
        warmup_steps = 100
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,  # weight_decay=0.0, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup_steps
        )
        return [optimizer], [scheduler]
