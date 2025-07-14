# 🧠 Image Classification using VGG16 (Transfer Learning)

This project implements an image classification pipeline using PyTorch and a pre-trained VGG16 model. The code uses transfer learning to classify grayscale images into multiple classes, based on a custom dataset structured in folders.

---

## 📁 Dataset Structure

The dataset should follow the format:

📂 dataset_root/
├── 📂 Training/
│ ├── 📂 Class1/
│ ├── 📂 Class2/
│ └── ...
└── 📂 Testing/
├── 📂 Class1/
├── 📂 Class2/
└── ...

Each class folder should contain the respective images.


 ⚙️ Requirements

Install the required libraries using:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

🚀 How to Run
Set the dataset paths in the script:

train_folder_path = "path/to/Training"
test_folder_path = "path/to/Testing"

Run the script:

  ```bash
python main.py
```
The script will:

Preprocess the data (resize, convert to 3 channels, normalize)

Train a VGG16 model for 10 epochs

Save and reload the trained model

Predict on the test set

Display a confusion matrix

Print the classification report

 Model Details
Backbone: VGG16 (pre-trained on ImageNet)

Modification: Final classifier layer adapted to number of classes

Input: Grayscale images resized to 224x224 and converted to 3-channel

Loss Function: CrossEntropyLoss

Optimizer: Adam

Output
Loss per epoch

Confusion matrix (displayed using Seaborn)

Classification report (precision, recall, F1-score for each class)

✨ Author
Shruthikha Suresh
Feel free to reach out for collaboration or questions!

