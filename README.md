# **Semantic Image Search Engine using CLIP**

## **📌 Project Overview**
This project builds a **semantic search engine for images**, allowing users to find pictures based on textual descriptions (e.g., *"a cat sitting on a chair"*). Instead of manually searching through images, the system enables users to **retrieve relevant images** by simply describing their contents.

This is achieved using **CLIP (Contrastive Language-Image Pretraining)**, a powerful multimodal model developed by **OpenAI**, which learns **joint representations of images and text**.

---

## **🖼️ How It Works?**
1. **Preprocess Images**: Load and preprocess images from the **MS-COCO dataset**.
2. **Generate Image Embeddings**: Extract feature vectors from images using **CLIP’s vision encoder**.
3. **Generate Text Embeddings**: Convert text queries into embeddings using **CLIP’s text encoder**.
4. **Perform Similarity Search**: Compute **cosine similarity** between the text embedding and all image embeddings.
5. **Retrieve and Display Results**: Return the **most relevant images** based on similarity scores.

---

## **🧠 Model Used: CLIP**
We use **OpenAI’s CLIP (Contrastive Language-Image Pretraining)**, specifically the **ViT-B/32** variant.

### **CLIP Model Variants**
| **Model**      | **Parameters** | **Image Encoder** | **Text Encoder** | **Speed vs. Accuracy** |
|---------------|--------------|------------------|----------------|------------------|
| **ViT-B/32**  | 151M         | ViT-Base, Patch 32 | Transformer     | Fast, decent accuracy |
| **ViT-B/16**  | 151M         | ViT-Base, Patch 16 | Transformer     | Slower, better accuracy |
| **ViT-L/14**  | 428M         | ViT-Large, Patch 14 | Transformer     | High accuracy, slower |
| **RN50**      | 102M         | ResNet-50         | Transformer     | Faster, less accuracy |

For this project, **ViT-B/32** is chosen as a balance between **speed and accuracy**.

---

## **📂 Dataset: MS-COCO 2017**
We use the **MS-COCO dataset** (Microsoft Common Objects in Context), which contains:
- **330K images** with real-world scenes.
- **Captions:** Each image has **five textual descriptions**.

**Download Links:**
- **[COCO 2017 Train Images (19GB)](http://images.cocodataset.org/zips/train2017.zip)**
- **[COCO 2017 Captions Annotations (23MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)**

---

## **🚀 Steps to Run the Project**

### **1️⃣ Install Dependencies**
Ensure you have Python **3.7+** and install the required libraries:
```bash
pip install torch torchvision clip-by-openai numpy scikit-learn matplotlib pillow requests
```

### **2️⃣ Download COCO Dataset**
You can manually download and extract images or use the following script:
```python
import requests, zipfile, os

# Create folder
os.makedirs("coco_dataset/annotations", exist_ok=True)
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
annotations_path = "coco_dataset/annotations_trainval2017.zip"

# Download and extract
with open(annotations_path, "wb") as f:
    f.write(requests.get(annotations_url).content)
with zipfile.ZipFile(annotations_path, "r") as zip_ref:
    zip_ref.extractall("coco_dataset/annotations")
os.remove(annotations_path)
```

### **3️⃣ Generate Image & Text Embeddings**
Run the script to **process images and captions**, then save embeddings:
```bash
python process_images.py
```

### **4️⃣ Perform Semantic Search**
To find images using text queries, run:
```python
from search import display_results
display_results("a dog running on the beach", top_k=5)
```

This will retrieve and display **top 5 matching images**.

---

## **🔍 Features & Optimizations**
✅ **Parallel Image Processing**: Uses `ThreadPoolExecutor` to speed up embedding generation.
✅ **Handles Corrupted Images**: Skips truncated/corrupt images to prevent crashes.
✅ **Efficient Search**: Uses **matrix multiplication (`np.dot`)** for fast similarity calculations.
✅ **Visual Results**: Displays matching images using Matplotlib.

---

## **📌 Future Improvements**
🔹 Use **FAISS** for even **faster image retrieval**.
🔹 Support **custom datasets** (upload personal images instead of COCO).
🔹 Implement a **web-based UI** using Streamlit or Flask.

---

## **👨‍💻 Contributors**
- **Hammad Khan** *(Lead Developer)*
- OpenAI (CLIP Model Developers)

---

## **📜 License**
This project is open-source and follows the **MIT License**.

🚀 **Happy Coding!** 🎉

