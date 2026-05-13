from PIL import Image, ImageDraw, ImageFont
import os

r50_path = 'qual_results/P2645_0005_r50.png'
mv4_path = 'qual_results/P2645_0005_mv4.png'

# Load images
r50 = Image.open(r50_path)
mv4 = Image.open(mv4_path)

gap = 10
label_h = 50

# Calculate canvas dimensions
W = r50.width + mv4.width + gap
H = r50.height + label_h

canvas = Image.new('RGB', (W, H), color=(255, 255, 255))
canvas.paste(r50, (0, label_h))
canvas.paste(mv4, (r50.width + gap, label_h))

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
except:
    font = ImageFont.load_default()

# Define labels
label_a = '(a) ResNet-50 baseline (74.53% mAP)'
label_b = '(b) MobileNetV4 distilled (64.80% mAP)'

# Calculate horizontal centers for each image
center_a = r50.width // 2
center_b = r50.width + gap + (mv4.width // 2)

# REPLACED: Compatibility-safe text width calculation
def get_text_width(text, font, draw_obj):
    # Try the older textsize method first for legacy support
    if hasattr(draw_obj, 'textsize'):
        return draw_obj.textsize(text, font=font)[0]
    # Fallback for even older versions
    return font.getsize(text)[0]

w_a = get_text_width(label_a, font, draw)
w_b = get_text_width(label_b, font, draw)

# Draw text using the manual center calculation
# This ensures it is truly centered, unlike the left-aligned look in P2645_comparison (3).jpg
draw.text((center_a - w_a // 2, 12), label_a, fill=(0,0,0), font=font)
draw.text((center_b - w_b // 2, 12), label_b, fill=(0,0,0), font=font)

# Save result
os.makedirs('qual_results', exist_ok=True)
out = 'qual_results/P2645_comparison_fixed.png'
canvas.save(out, dpi=(300, 300))
print(f'Saved to {out}')