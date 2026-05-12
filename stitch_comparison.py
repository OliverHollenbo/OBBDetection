from PIL import Image, ImageDraw, ImageFont
import os

r50_path  = 'qual_results/P2645_0005_r50.png'
mv4_path  = 'qual_results/P2645_0005_mv4.png'

r50 = Image.open(r50_path)
mv4 = Image.open(mv4_path)

gap       = 10
label_h   = 40
W         = r50.width * 2 + gap
H         = r50.height + label_h

canvas = Image.new('RGB', (W, H), color=(255, 255, 255))
canvas.paste(r50, (0, label_h))
canvas.paste(mv4, (r50.width + gap, label_h))

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
except:
    font = ImageFont.load_default()

draw.text((r50.width // 2, 6),         '(a) ResNet-50 baseline (74.53% mAP)',  fill=(0,0,0), font=font, anchor='mt')
draw.text((r50.width + gap + mv4.width // 2, 6), '(b) MobileNetV4 distilled (64.80% mAP)', fill=(0,0,0), font=font, anchor='mt')

os.makedirs('qual_results', exist_ok=True)
out = 'qual_results/P2645_comparison.png'
canvas.save(out, dpi=(300, 300))
print(f'Saved to {out}')
