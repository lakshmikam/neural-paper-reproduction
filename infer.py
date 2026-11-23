"""Evaluate a saved checkpoint without training. Example: python infer.py --case 5"""
import argparse
from pathlib import Path
import torch
from lagaris import Trial, grid, exact, CASES

p = argparse.ArgumentParser()
p.add_argument('--case', choices=CASES, default='5')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--results', type=Path, default=Path('results'))
args = p.parse_args()
state = torch.load(args.results/f'problem_{args.case}_seed_{args.seed}.pt', weights_only=True)
model = Trial(args.case)
model.load_state_dict(state['state_dict'])
z = grid(args.case, 30, offset=True)
print('Maximum absolute error:', float((model(z)-exact(args.case, z)).detach().abs().max()))
