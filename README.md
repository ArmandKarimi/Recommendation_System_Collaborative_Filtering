## 📌 README.md - MovieLens Recommendation System
```markdown
# 🎬 MovieLens Recommendation System

This is a Collaborative Filtering-based Recommendation System built with PyTorch. It predicts user ratings for movies and recommends the best-rated movies for a given user.

---

## 🚀 Features

✅ Uses MovieLens dataset for movie recommendations  
✅ Implements Collaborative Filtering (CF) with Embeddings
✅ Trains using PyTorch on CPU, CUDA, or Apple MPS (M1/M2 Macs)  
✅ RMSE Evaluation for model performance  
✅ Command-line user input for personalized recommendations  

---

## 📥 Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ArmandKarimi/Recommendation_System_Collaborative_Filtering.git
cd movielens-recommender
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv myenv
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate  # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎬 Setting Up Kaggle & Downloading MovieLens Data

### **Step 1: Install Kaggle API**
```bash
pip install kaggle
```

### **Step 2: Get Your Kaggle API Key**
1. Go to [Kaggle](https://www.kaggle.com/) and **sign in**.
2. Click on your profile picture (top-right) → **Account**.
3. Scroll down to the **API** section.
4. Click **Create New API Token** → This downloads a `kaggle.json` file.

### **Step 3: Place `kaggle.json` in the Correct Location**
#### **For macOS/Linux**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # Set file permissions
```
#### **For Windows**
Move `kaggle.json` to:
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

### **Step 4: Download MovieLens Dataset**
```bash
kaggle datasets download -d grouplens/movielens-20m-dataset --unzip -p ./data
```
- The dataset will be extracted to the `data/` folder.

---

## 🚀 Running the Model

### **Train & Evaluate  the Recommendation Model**
```bash
python main.py
```
Then, enter a **User ID** when prompted to return top 5 recommanded movies for the user.

---

## 🔥 Project Structure

```
.
├── README.md
├── main.py
├── project_structure.txt
├── requirements.txt
└── src
    ├── __pycache__
    ├── config.py
    ├── data
    └── utils
        ├── __init__.py
        ├── dataset.py
        ├── fetch_data.py
        ├── model.py
        ├── recommendation.py
        ├── test_model.py
        └── train_model.py
```
---

## 📊 Model Evaluation (RMSE)
To evaluate the trained model on the test set:
```bash
python main.py --evaluate
```
- RMSE (Root Mean Squared Error) is used to measure prediction accuracy.

---

## 🛠️ Troubleshooting

❓ **"ModuleNotFoundError: No module named 'dataset'"**  
✅ Ensure you are running the script from the project root folder:
```bash
cd movielens-recommender
python main.py
```

❓ **"No such file or directory: 'kaggle.json'"**  
✅ Make sure `kaggle.json` is in the **correct location** (`~/.kaggle/` or `C:\Users\<YourUsername>\.kaggle\`).

---

## 📜 License
This project is open-source and free to use. Feel free to contribute! 🚀


