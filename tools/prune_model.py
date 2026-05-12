"""
Structured channel pruning of MobileNetV4 backbone using DepGraph.
Fang et al., "DepGraph: Towards Any Structural Pruning", CVPR 2023.
"""
import argparse
import os
import torch
import torch_pruning as tp
from mmcv import Config
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Prune MobileNetV4 backbone')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--pruning-ratio', type=float, default=0.3,
                        help='channel pruning ratio (0.3 = remove 30% of channels)')
    parser.add_argument('--save-path', default='work_dirs/pruned_model.pth',
                        help='path to save pruned model')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and build model
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg,
                           test_cfg=cfg.test_cfg)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f'Original params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # Example input for DepGraph tracing
    example_input = torch.randn(1, 3, 1024, 1024)

    # Build dependency graph on backbone only
    # We prune the backbone feature extractor
    backbone = model.backbone.model  # timm model inside our wrapper

    ignored_layers = []
    # Ignore the final layer of each stage to preserve FPN input channels
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            ignored_layers.append(module)

    # Only keep first 60% of conv layers for pruning candidates
    # (prune earlier layers, preserve later ones that connect to FPN)
    prune_layers = ignored_layers[:int(len(ignored_layers) * 0.6)]
    ignored_layers = ignored_layers[int(len(ignored_layers) * 0.6):]

    # Set up DepGraph pruner
    imp = tp.importance.MagnitudeImportance(p=1)  # L1 norm importance
    pruner = tp.pruner.MagnitudePruner(
        backbone,
        example_inputs=torch.randn(1, 3, 256, 256),
        importance=imp,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored_layers,
    )

    # Apply pruning
    pruner.step()

    print(f'Pruned backbone params: {sum(p.numel() for p in backbone.parameters())/1e6:.2f}M')
    print(f'Total model params after pruning: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # Test forward pass
    try:
        with torch.no_grad():
            out = backbone(torch.randn(1, 3, 1024, 1024))
        print(f'Forward pass OK, {len(out)} feature maps')
        for i, o in enumerate(out):
            print(f'  Stage {i}: {o.shape}')
    except Exception as e:
        print(f'Forward pass failed: {e}')
        return

    # Save pruned model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'pruning_ratio': args.pruning_ratio,
    }, args.save_path)
    print(f'Saved pruned model to {args.save_path}')


if __name__ == '__main__':
    main()
