import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

random.seed(42)

words = open('sanskrit_names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
num_char = len(chars)
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)

d = 2
C = torch.randn((num_char + 1, d), generator=g)

W1 = torch.randn((block_size * d, 50), generator=g)
b1 = torch.randn(50, generator=g)
W2 = torch.randn((50, num_char + 1), generator=g)
b2 = torch.randn(num_char + 1, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
    
batch_size = 64

# print(sum(p.nelement() for p in parameters)) # prints total parameters
# lre = torch.linspace(-3, 0, 100000)
# lrs = 10 ** lre

# lri = []
# lossi = []
# stepi = []

lr = 10 ** -0.7
# lr = 0.1

for i in range(200000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)

    # forward pass
    emb = C[Xtr[ix]]
    # emb = torch.cat(torch.unbind(emb, 1), 1)
    # emb = emb.view(-1, block_size * d) # more efficient than above
    h =  torch.tanh(emb.view(-1, block_size * d) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    if i % 100000 == 0 and i > 0:
        lr = lr/10
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    # lossi.append(loss.log10().item())
    # stepi.append(i)

# plt.plot(lri, lossi)
# plt.show()

# plt.plot(stepi, lossi)
# plt.show()

@torch.nograd()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
plt.grid('minor')
plt.savefig('mlp_embeddings.png')

# sample from model
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h =  torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))