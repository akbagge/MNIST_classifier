import os
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import DictConfig
from mnist_classifier.data import corrupt_mnist
from mnist_classifier.model import MyAwesomeModel

@hydra.main(config_path="../../configs", config_name="config_train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    # Ensure correct working directory for data files
    os.chdir(hydra.utils.get_original_cwd())
    print(f"Current working directory reset to: {os.getcwd()}")

    # Resolve device
    DEVICE = (
        torch.device("cuda") if cfg.device == "cuda" and torch.cuda.is_available()
        else torch.device("mps") if cfg.device == "mps" and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {DEVICE}")

    # Initialize model and move to device
    model = MyAwesomeModel().to(DEVICE)

    # Load dataset
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size)

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training statistics
    statistics = {"train_loss": [], "train_accuracy": []}

    # Training loop
    print("Starting training...")
    for epoch in range(cfg.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item()}")

    print("Training complete")

    # Save the model
    torch.save(model.state_dict(), cfg.save_model_path)

    # Plot and save statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")

    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train Accuracy")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")

    fig.savefig(cfg.save_figure_path)
    print(f"Training statistics saved to {cfg.save_figure_path}")


if __name__ == "__main__":
    train()
