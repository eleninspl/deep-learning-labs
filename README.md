# Deep Learning Labs: Wide ResNets, MixUp and Pretrained Transformers

Two deep learning lab projects written as Google Colab notebooks in PyTorch and Hugging Face Transformers. The first trains Wide Residual Networks on CIFAR-10, compares depth, width and dropout, and tests whether MixUp augmentation makes the model more robust to corrupted images (CIFAR-10-C). The second fine-tunes DistilBERT for sentiment classification, then uses pretrained language models without further training on three multiple-choice benchmarks: PIQA, TruthfulQA and Winogrande.

The labs are my solutions to the lab projects of **Neural Networks and Deep Learning** (Νευρωνικά Δίκτυα και Βαθιά Μάθηση), an 8th-semester course at the School of Electrical and Computer Engineering, National Technical University of Athens (ECE NTUA), academic year 2024–25.

The repository also contains my written solutions to the two homework sets of **Machine Learning** (Μηχανική Μάθηση), a 7th-semester course at ECE NTUA in the same academic year. They are mostly derivations, with the Python code used for the numerical parts included in each PDF.

| Part | Files | Topic | Main tools |
|------|-------|-------|------------|
| [Lab 1](#lab-1-wide-resnets-and-mixup-on-cifar-10) | `lab1/NNDL25_labproject1_…ipynb` | Image classification and robustness to corruptions | PyTorch, torchvision, Wide ResNet, MixUp |
| [Lab 2](#lab-2-pretrained-transformers) | `lab2/NNDL25_labproject2_…ipynb` | Fine-tuning and zero-shot evaluation of language models | Transformers, Datasets, Sentence Transformers |
| [Homework 1](#machine-learning-homework-1) | `hwk1/ML24_hwk1_…pdf` | Least squares, Gaussians, Bayes classifier, MLP, kernels | NumPy, SciPy, Matplotlib |
| [Homework 2](#machine-learning-homework-2) | `hwk2/ML24_hwk2_…pdf` | Decision trees, k-means, hierarchical clustering, PCA, MDPs | NumPy, pandas, scikit-learn, SciPy |

All notebooks and reports are written in Greek. The code and printed results are mostly in English.

## Getting started

### Prerequisites

The notebooks were written for and run on **Google Colab with a GPU runtime** (an NVIDIA T4). The saved outputs come from these versions:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch / torchvision | 2.6.0 / 0.21.0 |
| matplotlib | 3.10.0 |
| transformers | 4.51.1 |
| datasets | 3.5.0 |
| evaluate | 0.4.3 |
| sentence-transformers | 3.4.1 |
| scikit-learn | 1.6.1 |

Each notebook installs what it needs in its first cells. Lab 1 also needs files that are not in this repository; see [Running lab 1](#running-lab-1).

```bash
git clone https://github.com/eleninspl/deep-learning-labs.git
cd deep-learning-labs
```

To run a notebook, upload it to Colab (File → Upload notebook) and select a GPU runtime. The notebooks are kept with their outputs, so you can read every result without running anything.

## Lab 1: Wide ResNets and MixUp on CIFAR-10

**Task.** Build an image classifier for CIFAR-10 (60,000 32×32 colour images in 10 classes) using Wide Residual Networks (Zagoruyko & Komodakis, 2016), starting from a template notebook provided by the course.

1. Train at least the three best depth/width combinations reported in the paper for CIFAR-10 with moderate data augmentation.
2. Train the same models again with dropout, as the paper suggests, and compare.
3. Take the best model, continue training it with a custom `Dataset` class, and evaluate it on CIFAR-10-C, a version of the CIFAR-10 test set with common image corruptions (noise, blur, weather, digital). Do this once without and once with MixUp, which the student implements inside the dataset class.
4. (Bonus) Compare the softmax confidence of the two models on their correct predictions.

**How it works.** To keep training within Colab limits, the template trains on a random 10% of CIFAR-10: 5,000 training images and 1,000 test images. Training uses random crops and horizontal flips, SGD with Nesterov momentum (lr 0.1, weight decay 5e-4), a cosine learning-rate schedule and 15 epochs.

The three architectures come from the paper's tables: WRN-28-10, WRN-40-4 and WRN-16-8. Each was trained with and without dropout 0.3:

| Model | Dropout | Test accuracy (1,000 images) |
|-------|---------|------------------------------|
| WRN-28-10 | – | 65.7% |
| WRN-28-10 | 0.3 | 58.4% |
| WRN-40-4 | – | 67.7% |
| WRN-40-4 | 0.3 | 59.4% |
| WRN-16-8 | – | 67.7% |
| WRN-16-8 | 0.3 | 63.6% |

With only 10% of the data and 15 epochs (the paper uses 200), the accuracies are far below the paper's, and dropout lowered accuracy for every model. WRN-16-8 and WRN-40-4 tied, which matches the paper's point that a shallow, wide network can match a deeper, thinner one.

For part 3, WRN-16-8 without dropout was trained for up to 5 more epochs from its saved weights. MixUp replaces an image and its one-hot label with a weighted mix of itself and a random other image: `x = λ·x₁ + (1−λ)·x₂`, `y = λ·y₁ + (1−λ)·y₂`, with λ drawn from Beta(0.2, 0.2). The model was evaluated on 10 of the 15 standard CIFAR-10-C corruption types, grouped into four categories, using 1,000 random images per corruption:

| Category | Without MixUp | With MixUp |
|----------|---------------|------------|
| Weather (snow) | 42.8% | 39.3% |
| Blur | 32.8% | 33.8% |
| Noise | 29.5% | 35.5% |
| Digital | 33.8% | 36.1% |
| **Average** | **34.7%** | **36.2%** |

For the bonus part, the model trained with MixUp was slightly less confident on its correct predictions on the clean CIFAR-10 test set (mean softmax 0.864 against 0.870). On CIFAR-10-C it was less confident on all 10 corruptions, for example 0.752 against 0.919 on contrast. The notebook plots both distributions as histograms and bar charts.

### Running lab 1

The notebook mounts Google Drive and expects the files from the course's assignment archive in a Drive folder:

- `wideresnet.py`, the Wide ResNet implementation provided with the assignment. It is imported as `WideResNet(depth, num_classes, widen_factor, dropRate)`. It is not included here; any implementation with the same constructor should work.
- `data/cifar/`: the CIFAR-10 Python batches (`cifar-10-batches-py/`). torchvision downloads these automatically.
- `data/cifar/CIFAR-10-C/`: the CIFAR-10-C `.npy` files and `labels.npy`. The dataset is publicly available from its authors (Hendrycks & Dietterich, 2019).

Set `PATH_TO_WORK_FOLDER` and the `sys.path.insert` line near the top of the notebook to that folder. The code calls `.cuda()` directly, so it needs a GPU.

## Lab 2: Pretrained transformers

**Part A: Fine-tuning.** Fine-tune a pretrained model of your choice on the Yelp Polarity sentiment dataset, using a balanced subset of 300 training and 300 test reviews. Try different hyperparameters, report the accuracy of each run, and explain the effect of the learning rate and batch size.

**How it works.** The notebook fine-tunes `distilbert-base-uncased` with the Hugging Face `Trainer`, AdamW and a linear learning-rate schedule. It runs eight configurations:

| Learning rate | Batch size | Epochs | Accuracy |
|---------------|------------|--------|----------|
| 5e-5 | 16 | 3 | 0.907 |
| 1e-4 | 16 | 3 | 0.890 |
| 2e-5 | 8 | 4 | 0.877 |
| 3e-5 | 32 | 3 | 0.873 |
| 1e-5 | 16 | 4 | 0.877 |
| 5e-5 | 32 | 2 | 0.873 |
| 3e-5 | 8 | 3 | 0.877 |
| 1e-4 | 32 | 4 | 0.877 |

The runs share one model, so each run continues from the weights of the previous one; see [Known limitations](#known-limitations).

**Part B: Using pretrained models on new tasks**, with no training.

| Benchmark | Task | Approach | Result |
|-----------|------|----------|--------|
| **PIQA** (100 random examples) | Choose the more sensible of two solutions to an everyday goal | Score `goal + solution` with a causal language model and pick the solution with the lower loss | gpt2 0.66, distilgpt2 0.65, gpt-neo-125M 0.66, gpt2-medium 0.67, **gpt-neo-1.3B 0.80** |
| **TruthfulQA** (first 100 questions with at least 2 correct answers) | Pick the best answer among the best answer, two correct answers and one incorrect answer. A correct answer counts only if its cosine similarity to the best answer is at least 0.95. | Same loss-based scoring with 3 causal LMs, combined with 6 sentence-transformer models for the similarity check | 0.21–0.28 across all 18 combinations. The language model matters much more than the similarity model. |
| **Winogrande** (100 random examples) | Fill the blank in a sentence with one of two names or nouns | Replace the blank with the mask token and compare the fill-mask scores of the two options | bert-base-uncased 0.56, **roberta-base 0.62**, distilroberta-base 0.50 |

Larger models did clearly better on PIQA, the clearest result of part B. TruthfulQA was hard for every small GPT-2 model.

### Running lab 2

Upload the notebook to Colab with a GPU runtime and run it top to bottom. It downloads the datasets and models from the Hugging Face Hub. The largest model, `EleutherAI/gpt-neo-1.3B`, is a 5.3 GB download. Newer versions of the `datasets` library no longer accept the short dataset names the notebook uses; see [Known limitations](#known-limitations).

## Machine Learning homework 1

Written solutions (`hwk1/ML24_hwk1_03121087_NasopoulouEleni.pdf`) to five exercises:

1. **Least squares estimation**: For a linear model with Gaussian noise, derive the distribution of the outputs and the least-squares estimator. Show that the estimator is unbiased and find its covariance, and show that the fitted values are a projection onto the column space of the design matrix.
2. **Bivariate Gaussian**: Find the joint and conditional densities, compute conditional probabilities, and plot an equal-density contour.
3. **Bayes classifier with three classes**: For three Gaussian classes with a shared covariance, compute the posteriors for a test point, derive the linear decision boundaries, plot 500 samples per class, and estimate the error rate for class ω₂ by simulation (about 2%).
4. **MLP for a three-class XOR**: Design by hand a network with step activations that computes `(x₁ + x₂) mod 3` for inputs in {0, 1, 2}. The solution uses 4 hidden and 3 output neurons and checks all 9 inputs in code.
5. **Kernels and SVMs**: Show that kernels are symmetric, that RBF feature vectors are at most √2 apart, and that an RBF SVM's output tends to its bias far from the training data.

## Machine Learning homework 2

Written solutions (`hwk2/ML24_hwk2_03121087_NasopoulouEleni.pdf`) to five exercises:

1. **Decision trees and random forests**: Build a Gini decision tree by hand, find a smaller tree with the same accuracy, and train a three-tree random forest.
2. **k-means**: Run k-means by hand on 8 points. Implement it from scratch and apply it to the Iris dataset: 89.3% success rate with all four features and 94.7% with petal length and width only.
3. **Hierarchical clustering**: Show that `1 − cos θ` is a dissimilarity measure but not a metric, then run single-linkage and complete-linkage clustering with it and draw the dendrograms.
4. **PCA**: Derive with Lagrange multipliers that the principal directions are the top eigenvectors of the covariance matrix, and find the fraction of variance they explain.
5. **Markov decision process**: Model a 4×4 grid world with a goal and a trap, write the Bellman equation, and compute the value of the start state under a policy that heads straight to the goal (1.219 for γ = 0.9).

## Known limitations

The notebooks are kept exactly as submitted. While writing this README, I reviewed them again and found the issues below. They do not stop the notebooks from showing the results above, but they matter if you reuse the code or compare the numbers.

**Lab 1**

- It depends on files from the course archive (`wideresnet.py`, CIFAR-10-C) and on hard-coded Google Drive paths, so it does not run on its own.
- The same 1,000 test images are used for the per-epoch "validation" printouts and for the final accuracy. In part 3, the best checkpoint is chosen by CIFAR-10-C accuracy and then reported on CIFAR-10-C. Each evaluation also draws a new random sample of 1,000 images per corruption. The MixUp gain (34.7% → 36.2%) is therefore smaller than the epoch-to-epoch variation in the logs (25%–34% for the run without MixUp).
- The part 3 training loop applies `nn.Softmax` to the outputs before `nn.CrossEntropyLoss`, which applies log-softmax itself, so softmax is applied twice. The loop also creates a cosine scheduler but never calls `scheduler.step()`, so the learning rate stays at 0.1. The last of the 5 epochs is never evaluated.
- MixUp is applied to images whose index is divisible by 5. That is always the same 20% of the training set, not the random one-in-five chance the code comment describes.
- In the part 1 table in the notebook, the "Loss" column is the sum of batch-mean losses divided by the number of images, about the mean loss divided by 32. It is not a percentage.
- In the bonus part, the per-corruption confidence table written in the notebook's markdown differs from the printed outputs in four values (impulse noise, motion blur and contrast). This README uses the printed outputs.

**Lab 2**

- GitHub cannot display the lab 2 notebook ("the 'state' key is missing from 'metadata.widgets'"). Colab and Jupyter open it normally.
- With current versions of `datasets` and `huggingface_hub`, the short dataset names (`yelp_polarity`, `piqa`, `truthful_qa`, `winogrande`) fail to load. The namespaced names `fancyzhx/yelp_polarity`, `truthfulqa/truthful_qa` and `allenai/winogrande` work. PIQA (`ybisk/piqa`) still uses a loading script, so it needs `datasets<4` and `trust_remote_code=True`.
- In part A, all eight configurations train the same model object, so each run starts from the weights of the run before it. The table is not a comparison of independent runs. The scheduler's step count also uses floor division, so the learning rate reaches zero slightly before training ends.
- In TruthfulQA, choosing the best answer itself counts as correct only when the same text also appears as one of the first two correct answers. This is true for 76 of the 100 questions. With a 0.95 threshold, the similarity check accepts almost only identical texts, which is why the six similarity models give nearly the same accuracy.
- In Winogrande, the fill-mask pipeline returns only its top 5 tokens. If neither option is among them, both scores are 0 and the prediction defaults to option 2. Option 2 is the right answer in 45 of the 100 examples, so a model that never finds either option still scores 0.45. Passing `targets=[option1, option2]` to the pipeline would score both options directly. The comment cell also says sequence-classification models were used, but the code uses fill-mask models.

## For students taking the course

This repository is here to help you understand the material: how the techniques are used in practice and what results to expect at this small scale. Write your own solutions. The Machine Learning homework sets state that solutions must be individual work, and copied work defeats the purpose of the labs. The limitations above are also a list of mistakes worth avoiding in your own code.

The original papers are the best reference for lab 1: *Wide Residual Networks* (Zagoruyko & Komodakis, 2016), *mixup: Beyond Empirical Risk Minimization* (Zhang et al., 2018) and *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (Hendrycks & Dietterich, 2019). For lab 2, the Hugging Face course and documentation cover every API used.
