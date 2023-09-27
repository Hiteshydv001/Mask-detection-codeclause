
# Mask Detection Using Machine Learning

This project is a Mask Detection system implemented using machine learning and computer vision techniques to identify whether a person in an image or video is wearing a mask (protective face covering) or not. The system contributes to safety and compliance monitoring, especially during situations that require mask-wearing.

**Introduction**

Mask detection is a critical application in situations that require adherence to mask-wearing guidelines, such as during a pandemic. This project demonstrates how to build, train, and evaluate a mask detection system using machine learning and computer vision.


## Dataset:-

The mask detection system was trained on a labeled dataset of images and videos containing people with and without masks. The dataset includes diverse scenarios, face orientations, lighting conditions, and mask types to ensure robustness.


## Demo

Insert gif or link to demo



## Roadmap:-

**Mask Detection Using Machine Learning: Major Steps**

**1. Data Collection:**  
Gather a labeled dataset of images or videos containing people with and without masks. Ensure the dataset is diverse, includes various face orientations, lighting conditions, and mask types.

**2. Data Preprocessing:**
Resize images to a consistent size to ensure compatibility with machine learning models. Normalize pixel values (typically between 0 and 1). Augment the data with techniques like rotation, scaling, and flipping to increase the dataset's size and robustness.

**3. Data Labeling:**
Ensure that each image or video frame is labeled as "with mask" or "without mask." Annotate the dataset accurately and consistently.

**4. Data Splitting:**
Divide the dataset into training, validation, and test sets (e.g., 70% for training, 15% for validation, 15% for testing).

**5. Model Selection:**
- Convolutional Neural Networks (CNNs)
- Transfer Learning (e.g., using pre-trained models like ResNet or MobileNet)
- Support Vector Machines (SVMs)
- Random Forests

**6. Model Training:**
Train the selected model on the training dataset. Fine-tune hyperparameters, such as learning rate, batch size, and network architecture. Implement techniques like early stopping and model checkpointing to prevent overfitting.

**7. Model Evaluation:**
Evaluate the trained model on the validation and test datasets using metrics like accuracy, precision, recall, F1-score, and ROC AUC. Check for false positives and false negatives, as they have different implications in mask detection.



## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Tech Stack

**Client:** Anaconda || Jupyter Notebook

**Server:** 


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)


