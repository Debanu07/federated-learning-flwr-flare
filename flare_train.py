import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import nvflare.client as flare
from model import SimpleNet

# ── Exact same hyperparameters as Flower experiment ───────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 20       
BATCH   = 32      # same
LR      = 0.01    # same
CLIENTS = 2       # same

# ── Absolute path so both sites find data regardless of CWD ──
DATA_DIR = os.path.expanduser("~/mnist_data")

# ── Load data — same split as Flower ──────────────────────
def load_data(client_id):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset  = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)  # ← fixed
    testset   = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)  # ← fixed
    partition = len(trainset) // CLIENTS
    indices   = list(range(client_id * partition, (client_id + 1) * partition))
    return (
        DataLoader(Subset(trainset, indices), batch_size=BATCH, shuffle=True),
        DataLoader(testset, batch_size=BATCH)
    )

# ── Same train function ────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        optimizer.step()

# ── Same evaluate function ─────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out     = model(images)
            loss   += criterion(out, labels).item()
            correct += (out.argmax(1) == labels).sum().item()
    return loss / len(loader), correct / len(loader.dataset)

# ── FLARE client logic ─────────────────────────────────────
def main():
    # initialize FLARE client
    flare.init()

    # get site name from FLARE → tells us which client we are
    site_name = flare.get_site_name()
    client_id = 0 if "site-1" in site_name else 1
    print(f"\n[{site_name}] Starting as Client {client_id}")

    # load local data
    train_loader, test_loader = load_data(client_id)
    print(f"[{site_name}] Loaded {len(train_loader.dataset)} training samples")

    # build model — same SimpleNet as Flower
    model     = SimpleNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # ── FL training loop ───────────────────────────────────
    # FLARE sends a FLModel each round — we train and send back
    while flare.is_running():

        # receive global model from FLARE server
        input_model = flare.receive()
        round_num   = input_model.current_round
        print(f"\n[{site_name}] Round {round_num + 1} started")

        if input_model.params:
            model.load_state_dict(input_model.params)

        # train locally — same as Flower fit()
        for epoch in range(EPOCHS):
            train(model, train_loader, optimizer, criterion)
            loss, acc = evaluate(model, test_loader, criterion)
            print(f"  Epoch {epoch + 1} → Accuracy: {acc:.4f}  Loss: {loss:.4f}")

        # evaluate final round accuracy
        loss, acc = evaluate(model, test_loader, criterion)
        print(f"[{site_name}] Round {round_num + 1} → Accuracy: {acc:.4f}  Loss: {loss:.4f}")

        # send updated weights back to FLARE server
        output_model = flare.FLModel(
            params=model.state_dict(),
            metrics={"accuracy": acc, "loss": loss},
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)}
        )
        flare.send(output_model)

    print(f"\n[{site_name}] Training complete")

if __name__ == "__main__":
    main()