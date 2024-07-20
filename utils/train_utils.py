import torch
import torch.nn as nn


# 计算学习率的方法
def get_lr(step, d_model, warmup_steps=4000):
    lr = (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    return lr


def train_model(model, train_dataloader, val_dataloader, epochs, pad_token, device):
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            # 自回归错位练
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            for param_group in optimizer.param_groups:
                param_group['lr'] = get_lr(epoch+1, model.d_model, 4000)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        # 验证
        if val_dataloader is not None:
            val_loss = valid_model(model, val_dataloader, pad_token, device)
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}')


# 验证过程
def valid_model(model, val_dataloader, pad_token, device):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader)
    model.train()
    return avg_loss
