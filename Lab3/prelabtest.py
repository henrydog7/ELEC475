from utils.dataset import PetNoseDataset
from models.snoutnet import *
model = checks[0][0]# SnoutNet (the loop leaves VGG16 in model)
dataset = PetNoseDataset(split="test")
model.eval()
image, label, orig_size = dataset[0]
with torch.no_grad():
# Model output is normalized: (u/227, v/227)
    pred = model(image.unsqueeze(0))[0]
print("Ground truth:", label / 227 * orig_size)
print("Prediction: ", pred * orig_size)