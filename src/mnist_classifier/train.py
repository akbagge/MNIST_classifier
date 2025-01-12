import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger


class MyAwesomeModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()  # Accuracy calculation

        # Logging scalar values
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()

        # Log validation loss and accuracy (once per epoch)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)



    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

def main():
    from pytorch_lightning.loggers import WandbLogger

    # Initialize WandB logger
    wandb_logger = WandbLogger(project="dtu_mlops")

    # Prepare the MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="data", download=True, transform=transform)
    train_data, val_test_data = random_split(dataset, [55000, 5000])
    val_data, test_data = random_split(val_test_data, [2500, 2500])  # Split into validation and test sets

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # Train the model
    model = MyAwesomeModel()
    trainer = pl.Trainer(max_epochs=3, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)

    # Test the model (optional)
    trainer.test(model, test_loader)



if __name__ == "__main__":
    main()
