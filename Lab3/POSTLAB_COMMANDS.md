# Lab 3 Post-Lab Commands

Run these commands from the `Lab3` directory. They assume the Oxford-IIIT
Pet Nose dataset is in `data/oxford-iiit-pet-noses/` and the provided
`snoutnet.pth`, `alexnet.pth`, and `vgg16.pth` weights are in `checkpoints/`.
The three first-stage prediction files are already in `predictions/`; if they
are missing, run the optional prediction command below before evaluating.

## 1. Set up the environment

```bash
cd Lab3
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p checkpoints plots predictions
```

## 2. Check the correction dataset and models

```bash
python -m utils.correction_dataset
python -m models.correction
```

Check `plots/correction_patches.png` and `plots/correction_patches_aug.png`:
the green target marker should be on the nose. The standard `CorrectionNet`
summary should show 766,594 parameters.

## 3. Train the four correction models

Each command trains for 20 epochs by default and saves the checkpoint and loss
plot needed for the report.

`CorrectionNetModified` uses four two-convolution residual blocks with
projection skips when channel counts change, 2x2 downsampling, and dropout
before the final output layer.

```bash
python train_correction.py --no-augment --save-model checkpoints/correction_noaug.pth --plot-file plots/loss_correction_noaug.png
python train_correction.py --save-model checkpoints/correction_aug.pth --plot-file plots/loss_correction_aug.png
python train_correction.py --modified --no-augment --save-model checkpoints/correction_mod_noaug.pth --plot-file plots/loss_correction_mod_noaug.png
python train_correction.py --modified --save-model checkpoints/correction_mod_aug.pth --plot-file plots/loss_correction_mod_aug.png
```

## 4. Correct the test predictions and evaluate

```bash
python correct.py --checkpoint checkpoints/correction_noaug.pth
python correct.py --checkpoint checkpoints/correction_aug.pth
python correct.py --modified --checkpoint checkpoints/correction_mod_noaug.pth
python correct.py --modified --checkpoint checkpoints/correction_mod_aug.pth
python evaluate.py
```

If any first-stage files (`predictions/snoutnet.pt`, `predictions/alexnet.pt`,
or `predictions/vgg16.pt`) are missing, generate them first:

```bash
python predict.py --model all
```

`evaluate.py` prints the accuracy and cost tables and updates `plots/ced.png`
and the best/worst prediction plots. Include those results, the four loss plots,
and your analysis in the post-lab report.

## 5. Package the submission

After saving your report as `Lab3/POSTLAB_REPORT.pdf`, run this from the
repository root and replace `STUDENT_NUMBER` with your student number (or the
two student numbers joined by an underscore). The archive excludes the dataset,
provided model weights, predictions, and local virtual environment.

```bash
zip -r ELEC475_Lab3_Submission.zip \
  Lab3/*.py \
  Lab3/models Lab3/utils Lab3/plots \
  Lab3/checkpoints/correction_noaug.pth \
  Lab3/checkpoints/correction_aug.pth \
  Lab3/checkpoints/correction_mod_noaug.pth \
  Lab3/checkpoints/correction_mod_aug.pth \
  Lab3/LBalog_PostLab3Report.pdf 
  Lab3/requirements.txt
```
