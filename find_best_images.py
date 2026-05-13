import pickle
import numpy as np

with open('/home/cv01f26/DATA/split_ss_dota1_0/val/annfiles/patch_annfile.pkl', 'rb') as f:
    data = pickle.load(f)

cls = data['cls']

fname_to_item = {item['filename']: item for item in data['content']}

results = []
for item in data['content']:
    labels = item['ann']['labels']
    n_total = len(labels)
    unique_classes = len(set(labels.tolist()))
    fname = item['filename']

    density_score = 1.0 if 10 <= n_total <= 40 else 0.5
    score = unique_classes * density_score

    results.append((score, unique_classes, n_total, fname))

results.sort(reverse=True)

print(f"\nTop 10 most visually rich patches:")
print(f"{'filename':<25} {'classes':>8} {'objects':>8}   class names")
print("-" * 80)
for score, n_cls, n_obj, fname in results[:10]:
    item = fname_to_item[fname]
    class_names = sorted(set(cls[l] for l in item['ann']['labels'].tolist()))
    print(f"{fname:<25} {n_cls:>8} {n_obj:>8}   {', '.join(class_names)}")

print("\nRun command:")
img_dir = '/home/cv01f26/DATA/split_ss_dota1_0/val/images/'
print("CUDA_VISIBLE_DEVICES=0 python visualize_comparison.py \\")
for _, _, _, fname in results[:10]:
    print(f"  {img_dir}{fname} \\")
