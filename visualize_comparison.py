import mmcv
import os
import sys
from mmdet.apis import init_detector, inference_detector

r50_config = 'configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py'
r50_ckpt   = 'work_dirs/oriented_rcnn_r50_bs2/epoch_12.pth'
mv4_config = 'configs/obb/oriented_rcnn/faster_rcnn_orpn_mobilenetv4_distill_24e_weighted_fpn_1x_dota10.py'
mv4_ckpt   = 'work_dirs/oriented_rcnn_mobilenetv4_distill_24e_weighted/epoch_24.pth'

images = sys.argv[1:]
if not images:
    print("Usage: python visualize_comparison.py img1.png img2.png ...")
    exit(1)

print('Loading ResNet-50...')
model_r50 = init_detector(r50_config, r50_ckpt, device='cuda:0')
print('Loading MobileNetV4...')
model_mv4 = init_detector(mv4_config, mv4_ckpt, device='cuda:0')

os.makedirs('qual_results', exist_ok=True)

for img_path in images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f'Running inference on {name}...')
    result_r50 = inference_detector(model_r50, img_path)
    result_mv4 = inference_detector(model_mv4, img_path)
    model_r50.show_result(img_path, result_r50, score_thr=0.3,
        out_file=f'qual_results/{name}_r50.png')
    model_mv4.show_result(img_path, result_mv4, score_thr=0.3,
        out_file=f'qual_results/{name}_mv4.png')
    print(f'  Saved qual_results/{name}_r50.png and {name}_mv4.png')

print('All done.')
