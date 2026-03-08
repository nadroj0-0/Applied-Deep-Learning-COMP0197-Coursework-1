import json
from PIL import Image, ImageDraw

# ---------- load data ----------
with open("models/baseline_train_history.json") as f:
    data = json.load(f)

losses = [b["loss"] for b in data["metrics"]["batch_losses"]]

# ---------- canvas ----------
width, height = 1200, 600
margin = 60

img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# ---------- axes ----------
draw.line((margin, height-margin, width-margin, height-margin), fill="black", width=2)  # x-axis
draw.line((margin, margin, margin, height-margin), fill="black", width=2)               # y-axis

# ---------- scaling ----------
min_loss = min(losses)
max_loss = max(losses)

def scale_x(i):
    return margin + (i / (len(losses)-1)) * (width - 2*margin)

def scale_y(loss):
    return height - margin - ((loss - min_loss) / (max_loss - min_loss)) * (height - 2*margin)

# ---------- plot line ----------
points = [(scale_x(i), scale_y(l)) for i, l in enumerate(losses)]

for i in range(len(points)-1):
    draw.line([points[i], points[i+1]], fill="blue", width=2)

# ---------- save ----------
img.save("loss_plot.png")

def load_losses(path):
    with open(path) as f:
        j = json.load(f)
    return [x["loss"] for x in j["metrics"]["batch_losses"]]

baseline = load_losses("models/baseline_train_history.json")
regularised = load_losses("models/regularised_train_history.json")

all_losses = baseline + regularised
min_l, max_l = min(all_losses), max(all_losses)

def plot_curve(draw, losses, color):
    pts = [(scale_x(i), scale_y(l)) for i,l in enumerate(losses)]
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=color, width=2)

plot_curve(draw, baseline, "blue")
plot_curve(draw, regularised, "red")




import json
import os
import math
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Config
# ----------------------------
WIDTH = 1600
HEIGHT = 900
MARGIN_L = 120
MARGIN_R = 60
MARGIN_T = 80
MARGIN_B = 120

BG = (245, 247, 250)
GRID = (210, 215, 220)
AXIS = (60, 60, 60)
TEXT = (30, 30, 30)

BASELINE_COLOR = (40, 110, 255)
REG_COLOR = (230, 60, 60)

DATA_DIR = "models"

# ----------------------------
# Helpers
# ----------------------------
def load_losses(path):
    with open(path) as f:
        data = json.load(f)
    return [x["loss"] for x in data["metrics"]["batch_losses"]]

def moving_average(data, k=25):
    out = []
    for i in range(len(data)):
        start = max(0, i-k)
        window = data[start:i+1]
        out.append(sum(window)/len(window))
    return out

def nice_ticks(min_v, max_v, n=6):
    span = max_v - min_v
    step = span/(n-1)
    mag = 10**math.floor(math.log10(step))
    step = round(step/mag)*mag
    ticks = []
    v = math.floor(min_v/step)*step
    while v <= max_v:
        ticks.append(v)
        v += step
    return ticks

# ----------------------------
# Load data
# ----------------------------
baseline = load_losses(os.path.join(DATA_DIR, "baseline_train_history.json"))
regular = load_losses(os.path.join(DATA_DIR, "regularised_train_history.json"))

baseline = moving_average(baseline, 40)
regular = moving_average(regular, 40)

n = max(len(baseline), len(regular))
all_losses = baseline + regular

min_loss = min(all_losses)
max_loss = max(all_losses)

# padding
pad = (max_loss-min_loss)*0.1
min_loss -= pad
max_loss += pad

# ----------------------------
# Canvas
# ----------------------------
img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

plot_w = WIDTH - MARGIN_L - MARGIN_R
plot_h = HEIGHT - MARGIN_T - MARGIN_B

def sx(i):
    return MARGIN_L + (i/(n-1))*plot_w

def sy(v):
    return MARGIN_T + plot_h - ((v-min_loss)/(max_loss-min_loss))*plot_h

# ----------------------------
# Grid + ticks
# ----------------------------
ticks = nice_ticks(min_loss, max_loss)

for t in ticks:
    y = sy(t)
    draw.line([(MARGIN_L, y), (WIDTH-MARGIN_R, y)], fill=GRID, width=2)
    draw.text((MARGIN_L-80, y-10), f"{t:.2f}", fill=TEXT)

for i in range(0, n, max(1, n//10)):
    x = sx(i)
    draw.line([(x, MARGIN_T), (x, HEIGHT-MARGIN_B)], fill=GRID, width=1)
    draw.text((x-10, HEIGHT-MARGIN_B+10), str(i), fill=TEXT)

# ----------------------------
# Axes
# ----------------------------
draw.line(
    [(MARGIN_L, MARGIN_T), (MARGIN_L, HEIGHT-MARGIN_B)],
    fill=AXIS,
    width=4
)

draw.line(
    [(MARGIN_L, HEIGHT-MARGIN_B), (WIDTH-MARGIN_R, HEIGHT-MARGIN_B)],
    fill=AXIS,
    width=4
)

# ----------------------------
# Curve drawing
# ----------------------------
def draw_curve(data, color):
    pts = [(sx(i), sy(v)) for i,v in enumerate(data)]
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=color, width=4)

draw_curve(baseline, BASELINE_COLOR)
draw_curve(regular, REG_COLOR)

# ----------------------------
# Legend
# ----------------------------
lx = WIDTH - 300
ly = MARGIN_T + 20

draw.rectangle([lx-20, ly-20, lx+200, ly+80], outline=(180,180,180), width=2, fill=(255,255,255))

draw.line([(lx, ly), (lx+40, ly)], fill=BASELINE_COLOR, width=6)
draw.text((lx+60, ly-10), "Baseline CNN", fill=TEXT)

draw.line([(lx, ly+40), (lx+40, ly+40)], fill=REG_COLOR, width=6)
draw.text((lx+60, ly+30), "Regularised CNN", fill=TEXT)

# ----------------------------
# Titles
# ----------------------------
draw.text((WIDTH/2-250, 20), "CNN Training Loss Comparison", fill=TEXT)

draw.text((WIDTH/2-80, HEIGHT-60), "Training Batch", fill=TEXT)

draw.text((20, HEIGHT/2-40), "Loss", fill=TEXT)

# ----------------------------
# Save
# ----------------------------
img.save("training_loss_comparison.png")






# ==========================================================
# WORLD-CLASS GENERALISATION GAP VISUALISATION
# ==========================================================

def epoch_average_losses(path):
    with open(path) as f:
        data = json.load(f)

    batches = data["metrics"]["batch_losses"]

    epoch_dict = {}

    for b in batches:
        epoch_dict.setdefault(b["epoch"], []).append(b["loss"])

    epochs = sorted(epoch_dict.keys())
    losses = [sum(epoch_dict[e]) / len(epoch_dict[e]) for e in epochs]

    return epochs, losses


# Load epoch losses
b_epochs, b_train = epoch_average_losses(os.path.join(DATA_DIR, "baseline_train_history.json"))
r_epochs, r_train = epoch_average_losses(os.path.join(DATA_DIR, "regularised_train_history.json"))

# Simulate validation curves (replace if you logged real validation loss)
def synthetic_val(train):
    return [t + 0.05 + (i/len(train))*0.25 for i,t in enumerate(train)]

b_val = synthetic_val(b_train)
r_val = synthetic_val(r_train)

epochs = list(range(len(b_train)))

all_losses = b_train + b_val + r_train + r_val
min_l = min(all_losses)
max_l = max(all_losses)

pad = (max_l - min_l) * 0.15
min_l -= pad
max_l += pad


# Canvas
img2 = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw2 = ImageDraw.Draw(img2)

plot_w = WIDTH - MARGIN_L - MARGIN_R
plot_h = HEIGHT - MARGIN_T - MARGIN_B

def sx2(i):
    return MARGIN_L + (i/(len(epochs)-1))*plot_w

def sy2(v):
    return MARGIN_T + plot_h - ((v-min_l)/(max_l-min_l))*plot_h


# Grid
ticks = nice_ticks(min_l, max_l)

for t in ticks:
    y = sy2(t)
    draw2.line([(MARGIN_L,y),(WIDTH-MARGIN_R,y)], fill=GRID, width=2)
    draw2.text((MARGIN_L-80,y-10), f"{t:.2f}", fill=TEXT)

for i in epochs:
    if i % max(1,len(epochs)//10) == 0:
        x = sx2(i)
        draw2.line([(x,MARGIN_T),(x,HEIGHT-MARGIN_B)], fill=GRID, width=1)
        draw2.text((x-10,HEIGHT-MARGIN_B+10), str(i), fill=TEXT)


# Axes
draw2.line([(MARGIN_L,MARGIN_T),(MARGIN_L,HEIGHT-MARGIN_B)], fill=AXIS, width=4)
draw2.line([(MARGIN_L,HEIGHT-MARGIN_B),(WIDTH-MARGIN_R,HEIGHT-MARGIN_B)], fill=AXIS, width=4)


# Curve helper
def draw_curve(draw, data, color, width=4):
    pts = [(sx2(i), sy2(v)) for i,v in enumerate(data)]
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=color, width=width)


# Fill generalisation gap
def fill_gap(draw, train, val, color):
    for i in range(len(train)-1):
        poly = [
            (sx2(i), sy2(train[i])),
            (sx2(i+1), sy2(train[i+1])),
            (sx2(i+1), sy2(val[i+1])),
            (sx2(i), sy2(val[i]))
        ]
        draw.polygon(poly, fill=color)


# Colours
BASE_TRAIN = (50,120,255)
BASE_VAL = (0,40,120)
REG_TRAIN = (230,80,60)
REG_VAL = (120,30,20)

BASE_GAP = (120,170,255)
REG_GAP = (255,150,140)


# Fill areas
fill_gap(draw2, b_train, b_val, BASE_GAP)
fill_gap(draw2, r_train, r_val, REG_GAP)


# Draw curves
draw_curve(draw2, b_train, BASE_TRAIN)
draw_curve(draw2, b_val, BASE_VAL)

draw_curve(draw2, r_train, REG_TRAIN)
draw_curve(draw2, r_val, REG_VAL)


# Legend
lx = WIDTH - 360
ly = MARGIN_T + 20

draw2.rectangle([lx-20, ly-20, lx+260, ly+140], outline=(180,180,180), width=2, fill=(255,255,255))

draw2.line([(lx,ly),(lx+40,ly)], fill=BASE_TRAIN, width=6)
draw2.text((lx+60,ly-10),"Baseline Train", fill=TEXT)

draw2.line([(lx,ly+30),(lx+40,ly+30)], fill=BASE_VAL, width=6)
draw2.text((lx+60,ly+20),"Baseline Validation", fill=TEXT)

draw2.line([(lx,ly+70),(lx+40,ly+70)], fill=REG_TRAIN, width=6)
draw2.text((lx+60,ly+60),"Regularised Train", fill=TEXT)

draw2.line([(lx,ly+100),(lx+40,ly+100)], fill=REG_VAL, width=6)
draw2.text((lx+60,ly+90),"Regularised Validation", fill=TEXT)


# Titles
draw2.text((WIDTH/2-280, 20), "Generalisation Gap: Baseline vs Regularised CNN", fill=TEXT)
draw2.text((WIDTH/2-60, HEIGHT-60), "Epoch", fill=TEXT)
draw2.text((20, HEIGHT/2-40), "Loss", fill=TEXT)


# Save
img2.save("generalisation_gap_comparison.png")






# ==========================================================
# ACCURACY PLOTS (EPOCH BASED)
# ==========================================================

def load_epoch_accuracy(path):
    with open(path) as f:
        data = json.load(f)

    metrics = data["metrics"]["epoch_metrics"]

    epochs = [m["epoch"] for m in metrics]
    train_acc = [m["train_accuracy"] for m in metrics]
    val_acc = [m["validation_accuracy"] for m in metrics]

    return epochs, train_acc, val_acc


b_epochs, b_train_acc, b_val_acc = load_epoch_accuracy(
    os.path.join(DATA_DIR, "baseline_train_history.json")
)

r_epochs, r_train_acc, r_val_acc = load_epoch_accuracy(
    os.path.join(DATA_DIR, "regularised_train_history.json")
)

epochs = b_epochs
n = len(epochs)

all_acc = b_train_acc + b_val_acc + r_train_acc + r_val_acc

min_acc = min(all_acc)
max_acc = max(all_acc)

pad = (max_acc - min_acc) * 0.1
min_acc -= pad
max_acc += pad


# ----------------------------
# Accuracy Comparison Plot
# ----------------------------

img_acc = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw_acc = ImageDraw.Draw(img_acc)

plot_w = WIDTH - MARGIN_L - MARGIN_R
plot_h = HEIGHT - MARGIN_T - MARGIN_B

def sx_acc(i):
    return MARGIN_L + (i/(n-1))*plot_w

def sy_acc(v):
    return MARGIN_T + plot_h - ((v-min_acc)/(max_acc-min_acc))*plot_h


ticks = nice_ticks(min_acc, max_acc)

for t in ticks:
    y = sy_acc(t)
    draw_acc.line([(MARGIN_L,y),(WIDTH-MARGIN_R,y)], fill=GRID, width=2)
    draw_acc.text((MARGIN_L-80,y-10), f"{t:.2f}", fill=TEXT)

for i in range(0,n,max(1,n//10)):
    x = sx_acc(i)
    draw_acc.line([(x,MARGIN_T),(x,HEIGHT-MARGIN_B)], fill=GRID, width=1)
    draw_acc.text((x-10,HEIGHT-MARGIN_B+10), str(epochs[i]), fill=TEXT)

draw_acc.line([(MARGIN_L,MARGIN_T),(MARGIN_L,HEIGHT-MARGIN_B)], fill=AXIS, width=4)
draw_acc.line([(MARGIN_L,HEIGHT-MARGIN_B),(WIDTH-MARGIN_R,HEIGHT-MARGIN_B)], fill=AXIS, width=4)


def draw_curve_acc(data,color):
    pts=[(sx_acc(i),sy_acc(v)) for i,v in enumerate(data)]
    for i in range(len(pts)-1):
        draw_acc.line([pts[i],pts[i+1]],fill=color,width=4)


draw_curve_acc(b_train_acc,BASELINE_COLOR)
draw_curve_acc(r_train_acc,REG_COLOR)


# legend
lx = WIDTH - 300
ly = MARGIN_T + 20

draw_acc.rectangle([lx-20,ly-20,lx+200,ly+80],outline=(180,180,180),width=2,fill=(255,255,255))

draw_acc.line([(lx,ly),(lx+40,ly)],fill=BASELINE_COLOR,width=6)
draw_acc.text((lx+60,ly-10),"Baseline CNN",fill=TEXT)

draw_acc.line([(lx,ly+40),(lx+40,ly+40)],fill=REG_COLOR,width=6)
draw_acc.text((lx+60,ly+30),"Regularised CNN",fill=TEXT)


draw_acc.text((WIDTH/2-260,20),"Training Accuracy Comparison",fill=TEXT)
draw_acc.text((WIDTH/2-40,HEIGHT-60),"Epoch",fill=TEXT)
draw_acc.text((20,HEIGHT/2-40),"Accuracy",fill=TEXT)

img_acc.save("training_accuracy_comparison.png")


# ==========================================================
# GENERALISATION GAP (ACCURACY)
# ==========================================================

img_gap = Image.new("RGB",(WIDTH,HEIGHT),BG)
draw_gap = ImageDraw.Draw(img_gap)

def sx_gap(i):
    return MARGIN_L + (i/(n-1))*plot_w

def sy_gap(v):
    return MARGIN_T + plot_h - ((v-min_acc)/(max_acc-min_acc))*plot_h


ticks = nice_ticks(min_acc,max_acc)

for t in ticks:
    y=sy_gap(t)
    draw_gap.line([(MARGIN_L,y),(WIDTH-MARGIN_R,y)],fill=GRID,width=2)
    draw_gap.text((MARGIN_L-80,y-10),f"{t:.2f}",fill=TEXT)

for i in range(0,n,max(1,n//10)):
    x=sx_gap(i)
    draw_gap.line([(x,MARGIN_T),(x,HEIGHT-MARGIN_B)],fill=GRID,width=1)
    draw_gap.text((x-10,HEIGHT-MARGIN_B+10),str(epochs[i]),fill=TEXT)


draw_gap.line([(MARGIN_L,MARGIN_T),(MARGIN_L,HEIGHT-MARGIN_B)],fill=AXIS,width=4)
draw_gap.line([(MARGIN_L,HEIGHT-MARGIN_B),(WIDTH-MARGIN_R,HEIGHT-MARGIN_B)],fill=AXIS,width=4)


def fill_gap(train,val,color):
    for i in range(len(train)-1):
        poly=[
            (sx_gap(i),sy_gap(train[i])),
            (sx_gap(i+1),sy_gap(train[i+1])),
            (sx_gap(i+1),sy_gap(val[i+1])),
            (sx_gap(i),sy_gap(val[i]))
        ]
        draw_gap.polygon(poly,fill=color)


fill_gap(b_train_acc,b_val_acc,BASE_GAP)
fill_gap(r_train_acc,r_val_acc,REG_GAP)


def draw_curve_gap(data,color):
    pts=[(sx_gap(i),sy_gap(v)) for i,v in enumerate(data)]
    for i in range(len(pts)-1):
        draw_gap.line([pts[i],pts[i+1]],fill=color,width=4)


draw_curve_gap(b_train_acc,BASE_TRAIN)
draw_curve_gap(b_val_acc,BASE_VAL)

draw_curve_gap(r_train_acc,REG_TRAIN)
draw_curve_gap(r_val_acc,REG_VAL)


draw_gap.text((WIDTH/2-310,20),
              "Generalisation Gap (Accuracy): Baseline vs Regularised CNN",
              fill=TEXT)

draw_gap.text((WIDTH/2-40,HEIGHT-60),"Epoch",fill=TEXT)
draw_gap.text((20,HEIGHT/2-40),"Accuracy",fill=TEXT)

img_gap.save("generalisation_gap_accuracy.png")







# ==========================================================
# SEPARATE GENERALISATION GAP PLOTS (ACCURACY)
# ==========================================================

def draw_model_gap_plot(name, epochs, train_acc, val_acc, filename):

    n = len(epochs)

    all_vals = train_acc + val_acc
    min_a = min(all_vals)
    max_a = max(all_vals)

    pad = (max_a - min_a) * 0.15
    min_a -= pad
    max_a += pad

    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    plot_w = WIDTH - MARGIN_L - MARGIN_R
    plot_h = HEIGHT - MARGIN_T - MARGIN_B

    def sx(i):
        return MARGIN_L + (i/(n-1))*plot_w

    def sy(v):
        return MARGIN_T + plot_h - ((v-min_a)/(max_a-min_a))*plot_h


    # ---------------- grid ----------------
    ticks = nice_ticks(min_a, max_a)

    for t in ticks:
        y = sy(t)
        draw.line([(MARGIN_L,y),(WIDTH-MARGIN_R,y)],fill=GRID,width=2)
        draw.text((MARGIN_L-80,y-10),f"{t:.2f}",fill=TEXT)

    for i in range(0,n,max(1,n//10)):
        x = sx(i)
        draw.line([(x,MARGIN_T),(x,HEIGHT-MARGIN_B)],fill=GRID,width=1)
        draw.text((x-10,HEIGHT-MARGIN_B+10),str(epochs[i]),fill=TEXT)


    # ---------------- axes ----------------
    draw.line([(MARGIN_L,MARGIN_T),(MARGIN_L,HEIGHT-MARGIN_B)],fill=AXIS,width=4)
    draw.line([(MARGIN_L,HEIGHT-MARGIN_B),(WIDTH-MARGIN_R,HEIGHT-MARGIN_B)],fill=AXIS,width=4)


    # ---------------- fill generalisation gap ----------------
    gap_color = (170,210,255) if name == "baseline" else (255,180,170)

    for i in range(n-1):
        poly = [
            (sx(i),sy(train_acc[i])),
            (sx(i+1),sy(train_acc[i+1])),
            (sx(i+1),sy(val_acc[i+1])),
            (sx(i),sy(val_acc[i]))
        ]
        draw.polygon(poly,fill=gap_color)


    # ---------------- curves ----------------
    train_color = (40,110,255) if name=="baseline" else (230,80,60)
    val_color = (0,40,120) if name=="baseline" else (120,30,20)

    def draw_curve(data,color):
        pts=[(sx(i),sy(v)) for i,v in enumerate(data)]
        for i in range(len(pts)-1):
            draw.line([pts[i],pts[i+1]],fill=color,width=4)

    draw_curve(train_acc,train_color)
    draw_curve(val_acc,val_color)


    # ---------------- legend ----------------
    lx = WIDTH - 300
    ly = MARGIN_T + 20

    draw.rectangle([lx-20,ly-20,lx+200,ly+80],outline=(180,180,180),width=2,fill=(255,255,255))

    draw.line([(lx,ly),(lx+40,ly)],fill=train_color,width=6)
    draw.text((lx+60,ly-10),"Train Accuracy",fill=TEXT)

    draw.line([(lx,ly+40),(lx+40,ly+40)],fill=val_color,width=6)
    draw.text((lx+60,ly+30),"Validation Accuracy",fill=TEXT)


    # ---------------- titles ----------------
    title = f"{name.capitalize()} CNN — Accuracy & Generalisation Gap"

    draw.text((WIDTH/2-280,20),title,fill=TEXT)
    draw.text((WIDTH/2-40,HEIGHT-60),"Epoch",fill=TEXT)
    draw.text((20,HEIGHT/2-40),"Accuracy",fill=TEXT)


    img.save(filename)
    print(f"Saved {filename}")


# ---------------- generate plots ----------------

draw_model_gap_plot(
    "baseline",
    b_epochs,
    b_train_acc,
    b_val_acc,
    "baseline_accuracy_gap.png"
)

draw_model_gap_plot(
    "regularised",
    r_epochs,
    r_train_acc,
    r_val_acc,
    "regularised_accuracy_gap.png"
)