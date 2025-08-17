# Multi-Task Learning for Image Classification

This project implements a **multi-task learning (MTL)** model that performs **image classification** and **color recognition** simultaneously. The model is trained on a dataset of clothing images from Kaggle ([Clothes Recommendation System using DenseNet121](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)), and optimized to jointly minimize the classification and auxiliary color prediction losses.  

The approach leverages the fact that related tasks share useful representations, allowing the model to generalize better than training each task in isolation.  

---

## 🚀 Methodology

1. **Data Preprocessing**
   - Dataset: Clothing image dataset from the linked Kaggle notebook.
   - Images are loaded and transformed into tensors.
   - Data augmentation (resize, normalization, flips, etc.) is applied to improve robustness.
   - Labels are split into two categories:
     - **Class label** – the main classification task.
     - **Color label** – auxiliary attribute prediction.
   - **Note on Class Imbalance:**  
     Even after removing all classes with fewer than 8 instances, there remains a significant class imbalance. To address this, **class weights** were calculated based on the frequency of each class and saved to `class_weights.pt`. These weights were then applied during training to reduce bias toward majority classes. This adjustment helped improve fairness across tasks.

2. **Model Architecture**
   - **Transfer Learning with Convolutional Backbone:**  
     The model uses a pretrained convolutional neural network (CNN) backbone (transfer learning) to extract feature representations from images. CNNs are highly effective for visual tasks because they:
       - Learn local patterns through convolutional filters.
       - Share parameters across the image, reducing complexity.
       - Build hierarchical features, from edges and textures in early layers to complex shapes in deeper layers.
     - Transfer learning allows us to leverage knowledge from large-scale pretrained models, which speeds up convergence and improves performance, especially when the dataset is relatively small.
   - **Two Task-Specific Heads:**
     - **Classification head** for the main class prediction.
     - **Color head** for predicting image color.

3. **Training Setup**
   - Loss = weighted sum of classification loss + color loss.
   - Optimizer: Adam / SGD (as configured in the notebook).
   - Metrics:
     - **Training loss & accuracy**
     - **Validation accuracy** for both tasks

4. **Results**
   - Best validation classification accuracy: **76.12%**
   - Validation color accuracy: **62.68%**
   - Model checkpoint saved as: `image_mtl_best.pt`
   - Accuracy values should be interpreted with caution due to class imbalance, even after applying weighted loss adjustment.

---

## 📊 Training Logs (Excerpt)

```
Epoch 3:
step 200: loss=1.7382 acc=73.89%
step 400: loss=1.6900 acc=74.16%
step 600: loss=1.7151 acc=73.82%
step 800: loss=1.7112 acc=73.83%
train acc=74.08% | val cls=76.12% | val color=62.68%
✅ saved image_mtl_best.pt
```

---

## Example Output

<img width="1000" height="1000" alt="Screenshot 2025-08-17 at 6 14 48 PM" src="https://github.com/user-attachments/assets/cfb70746-11a1-440b-bd49-e8dfeaf749ea" />
<img width="1000" height="1000" alt="Screenshot 2025-08-17 at 6 14 45 PM" src="https://github.com/user-attachments/assets/647a9fca-e8e1-4df1-828a-4da76ed8c6da" />


---

## 🛠️ How to Run

### 1. Run on Kaggle
1. Open the provided notebook (`model-training.ipynb`) in **Kaggle Notebooks**.
2. Attach the dataset from [this link](https://www.kaggle.com/code/rahmaezzat66/clothes-recommendation-system-using-densenet121) in the **Dataset panel**.
3. Click **Run All** – the notebook will:
   - Preprocess the data
   - Train the MTL model
   - Save the best checkpoint (`image_mtl_best.pt`) in the output directory

You can then download the trained model from **Kaggle’s output tab**.

---

### 2. Run Locally

#### Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install torch torchvision matplotlib scikit-learn
  ```
  (plus any dataset-specific libraries mentioned in the notebook)

#### Steps
1. Clone this repository and move into the project directory:
   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```
2. Download and extract the dataset locally from the Kaggle link.
3. Place your dataset in the `data/` folder (or update paths in the notebook).
4. Open the notebook locally:
   ```bash
   jupyter notebook model-training.ipynb
   ```
5. Run all cells to start training.
6. The best model will be saved as:
   ```
   ./image_mtl_best.pt
   ```

---

## 📦 Repository Structure
```
├── model-training.ipynb   # Main notebook
├── README.md              # Project documentation
├── data/                  # Dataset (not included, add locally)
├── image_mtl_best.pt      # Saved model checkpoint (after training)
├── class_weights.pt       # Saved class weights for imbalance handling
```

---

## 📌 Notes
- Multi-task learning improves classification by leveraging auxiliary supervision from the color prediction task.
- CNN backbones with transfer learning enable the model to extract powerful general-purpose features that benefit both tasks.
- Despite relatively high validation accuracies, performance metrics are influenced by dataset imbalance. Weighted loss adjustments were used to mitigate this.
- You can fine-tune hyperparameters (learning rate, epochs, loss weights) inside the notebook to further optimize performance.
