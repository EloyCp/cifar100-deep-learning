# CIFAR-100 Image Classification with Deep Learning

> **Master's Project** — Nuclio Digital School · Deep Learning module · Group 3

This project compares two different strategies for image classification on the **CIFAR-100** dataset (100 classes, 60,000 32×32 RGB images): one based on **Transfer Learning with pre-trained networks**, and another based on a **CNN trained from scratch**. The goal is to study which approach delivers the most reliable model and to document, experiment by experiment, why each design decision was made.

---

## Project Overview

The CIFAR-100 dataset is a well-known benchmark with 100 fine-grained categories. It is challenging because the images are very small (32×32 pixels) and the inter-class similarity is high. Two strategies are designed, trained, evaluated and compared.

### Strategy 1 — Pre-trained networks (Transfer Learning + Fine Tuning)

Networks pre-trained on ImageNet are reused as feature extractors and progressively adapted to CIFAR-100. Two architectures are benchmarked first (**VGG16** and **EfficientNetB0**), and the most promising one is then refined through:

- Fine Tuning with different numbers of frozen blocks (6/7, 5/7, 4/7, 3/7)
- Top model restructuring (number of dense layers and neurons)
- Regularization techniques (L2, Dropout, Batch Normalization)
- Data Augmentation
- Combinations of the above

### Strategy 2 — Custom CNN from scratch

A 3-block convolutional architecture is designed and iteratively refined:

- **Model 1 — Base** (Dropout 0.25)
- **Model 2 — Stronger regularization** (Dropout 0.5)
- **Model 3 — Hyperparameter tuning** (L1/L2 weight regularization at several scales, aggressive Dropout, learning-rate adjustment)
- **Model 4 — Data Augmentation** (geometric and photometric)

---

## Results Summary

| Strategy | Best Configuration | Test Accuracy | Notes |
|----------|--------------------|---------------|-------|
| Transfer Learning (VGG16) | Frozen base + dense top | ~35% | Heavy overfitting |
| Transfer Learning (EfficientNetB0) | Frozen base + dense top | ~34% | Stable, no overfitting |
| Fine Tuning (EfficientNetB0) | 3/7 blocks frozen | ~52% | Best fine-tuning result |
| Regularization on top model | Dropout 0.75 | ~55% | Best Strategy 1 result |
| **CNN from scratch (Model 2)** | **Dropout 0.5** | **~59%** | **Highest raw accuracy** |
| **CNN from scratch (Model 4.3)** | **Full Data Augmentation** | **~46%** | **Best generalization** |

### Key takeaway

A higher accuracy figure does not necessarily mean a better model. The CNN from scratch with full Data Augmentation reaches only 46% accuracy, but it is the only configuration where the training and validation curves **fully converge**, completely closing the overfitting gap. This makes it the most reliable choice for real-world conditions with varying lighting, framing or noise.

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** — model building, training and evaluation
- **NumPy** — numerical computations
- **Matplotlib** — visualization of samples and learning curves
- **scikit-learn** — preprocessing (`LabelBinarizer`, `train_test_split`) and evaluation (`classification_report`)

---

## Repository Structure

```
cifar100-deep-learning/
├── cifar100_classification.py     # Main script (full experimentation flow)
├── original_notebook/
│   └── Actividad1_DL_Grupo_3.ipynb  # Original Google Colab notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Files excluded from version control
```

---

## How to Run

> **Heads-up:** training the full pipeline is computationally heavy (several pre-trained networks plus around a dozen CNNs from scratch with 20–50 epochs each). A GPU is strongly recommended. Running it on a CPU is possible but very slow.

### 1. Clone the repository

```bash
git clone https://github.com/EloyCp/cifar100-deep-learning.git
cd cifar100-deep-learning
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the script

```bash
python cifar100_classification.py
```

The CIFAR-100 dataset is downloaded automatically by Keras the first time the script runs. Trained models are saved into a local `saved_models/` folder.

### Run it on Google Colab

The original notebook (`original_notebook/Actividad1_DL_Grupo_3.ipynb`) can be uploaded to [Google Colab](https://colab.research.google.com) for an interactive run with free GPU access.

---

## Methodology

Each experiment in this project follows the same logic:

1. **Define** a model variation with a clear hypothesis (e.g. *"raising Dropout to 0.5 should reduce overfitting"*).
2. **Train** it for the same number of epochs and with the same data split.
3. **Plot** the training and validation curves to compare overfitting and convergence speed.
4. **Document** the observed result and whether it confirms or refutes the hypothesis.
5. **Decide** the next experiment based on what was learned.

This iterative method makes the development reproducible and turns every "failed" experiment into useful information, since it narrows down the design space.

---

## License

This project was developed for educational purposes within the Master's program at Nuclio Digital School.
