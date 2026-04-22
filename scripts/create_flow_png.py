from PIL import Image, ImageDraw, ImageFont

width, height = 900, 650
img = Image.new('RGB', (width, height), 'white')
d = ImageDraw.Draw(img)
font = ImageFont.load_default()

# Utility for centered text

def centered_text(x, y, text, font, fill='black'):
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text((x - w / 2, y - h / 2), text, font=font, fill=fill)

# Draw title card
d.rectangle([50, 40, 850, 170], outline='#B4B2A9', width=2)
centered_text(450, 70, 'Bad PI coordinator', font, fill='#1a1a1a')

# Components
components = [
    ('Belief engine', 'updates hypothesis posteriors', 120),
    ('Hypothesis registry', 'gates LLM proposals', 340),
    ('Program writer', 'rewrites shared program.md', 560),
]
for label, note, x in components:
    d.rectangle([x - 80, 100, x + 80, 180], fill='#E8F1FF', outline='#1E88E5', width=2)
    centered_text(x, 130, label, font, fill='#1a1a1a')
    centered_text(x, 155, note, font, fill='#555555')

# Shared store
store_y = 250
d.rectangle([220, store_y, 680, store_y + 90], fill='#E8F5E9', outline='#2E7D32', width=2)
centered_text(450, store_y + 30, 'Shared experiment store', font, fill='#1a1a1a')
centered_text(450, store_y + 60, 'program.md · configs · results · hypothesis state', font, fill='#555555')

# Workers
workers = [(115, 360, 'Worker 1'), (350, 360, 'Worker 2'), (585, 360, 'Worker N')]
for x, y, label in workers:
    d.rectangle([x - 55, y, x + 55, y + 70], fill='#FFF8E1', outline='#F9A825', width=2)
    centered_text(x, y + 28, label, font, fill='#1a1a1a')
    centered_text(x, y + 50, '5-minute experiment', font, fill='#555555')

# LLM proposer
d.rectangle([430, 410, 770, 500], fill='#F3E5F5', outline='#6A1B9A', width=2)
centered_text(600, 440, 'LLM proposer', font, fill='#1a1a1a')
centered_text(600, 465, 'suggests new hypotheses', font, fill='#555555')

# Autoresearch run box
d.rectangle([80, 250, 200, 330], fill='#FFEBEE', outline='#C62828', width=2)
centered_text(140, 285, 'train.py / prepare.py', font, fill='#1a1a1a')
centered_text(140, 310, '5-min autoresearch run', font, fill='#555555')

# Connectors
def arrow(x1, y1, x2, y2):
    d.line([x1, y1, x2, y2], fill='#374151', width=3)
    d.line([x2, y2, x2 - 8, y2 - 12], fill='#374151', width=3)
    d.line([x2, y2, x2 + 8, y2 - 12], fill='#374151', width=3)

arrow(120, 180, 120, 250)
arrow(350, 180, 350, 250)
arrow(580, 180, 580, 250)
centered_text(120, 220, 'update beliefs', font, fill='#424242')
centered_text(350, 220, 'sync hypothesis state', font, fill='#424242')
centered_text(580, 220, 'publish program.md', font, fill='#424242')

arrow(250, 295, 120, 360)
arrow(370, 295, 350, 360)
arrow(560, 295, 585, 360)
centered_text(230, 330, 'worker pulls config', font, fill='#424242')
centered_text(450, 330, 'worker pulls config', font, fill='#424242')
centered_text(590, 330, 'worker pulls config', font, fill='#424242')

arrow(115, 430, 115, 510)
arrow(350, 430, 350, 510)
arrow(585, 430, 585, 510)
centered_text(115, 540, 'trained model & metrics', font, fill='#555555')
centered_text(350, 540, 'trained model & metrics', font, fill='#555555')
centered_text(585, 540, 'trained model & metrics', font, fill='#555555')

arrow(115, 510, 350, 280)
arrow(350, 510, 350, 280)
arrow(585, 510, 540, 280)
centered_text(350, 360, 'results', font, fill='#424242')

arrow(600, 410, 600, 300)
centered_text(640, 325, 'structured proposals', font, fill='#424242')

img.save('docs/autoresearch_flow.png')
print('created docs/autoresearch_flow.png')
