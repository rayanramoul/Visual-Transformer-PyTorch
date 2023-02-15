


def train_loop():
    for  epoch in tqdm(range(N_EPOCHS),  desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            train_loss += loss.detach().cpu().item() / len(train_loader)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} train loss: {train_loss:.4f}")