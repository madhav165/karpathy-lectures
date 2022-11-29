import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('sanskrit_names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
num_char = len(chars)
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# training set of bigrams
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes=num_char+1).float()
yenc = F.one_hot(ys, num_classes=num_char+1).float()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((num_char+1, num_char+1), generator=g, requires_grad=True) # first num represents the dimension of data and second number of neurons

for k in range(500):
    #forward pass
    logits = xenc @ W # @ is a matrix multiplication operator in Pytorch. Intepreted as log counts for bigrams
    # softmax steps
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(xs.shape[0]), ys].log().mean() + 0.01 * (W**2).mean()

    #backward pass
    W.grad = None
    loss.backward()

    # update weights
    W.data += -10 * W.grad

    #forward pass
    logits = xenc @ W # @ is a matrix multiplication operator in Pytorch. Intepreted as log counts for bigrams
    # softmax steps
    counts = logits.exp() # equivalent to N
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(xs.shape[0]), ys].log().mean()

print(f'{loss.item()=}')

    
# sample from the neural net model
for i in range(20):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=num_char+1).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break
    print(''.join(out))