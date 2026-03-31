import argparse
import os
import torch
from model import GPT

parser = argparse.ArgumentParser(description='Generate text from a trained nanoGPT model')
parser.add_argument('--checkpoint', type=str,
                    default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint.pt'))
parser.add_argument('--prompt', type=str, default='', help='starting text (empty = unconditional)')
parser.add_argument('--tokens', type=int, default=500, help='number of tokens to generate')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
chars = ckpt['chars']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = GPT(
    vocab_size=ckpt['vocab_size'],
    n_embd=ckpt['n_embd'],
    n_head=ckpt['n_head'],
    n_layer=ckpt['n_layer'],
    block_size=ckpt['block_size'],
    dropout=0.0,
    device=args.device,
)
model.load_state_dict(ckpt['model_state_dict'])
model.to(args.device)
model.eval()

if args.prompt:
    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=args.device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=args.tokens)
    print(decode(output[0].tolist()))
