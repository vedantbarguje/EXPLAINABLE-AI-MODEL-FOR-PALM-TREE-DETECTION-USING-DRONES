# 🌴 Explainable AI for Palm Tree Detection using Drones

This project presents an interpretable deep learning system for palm tree detection from drone imagery. By combining state-of-the-art object detection models (Faster R-CNN with ResNet-50 FPN) and Explainable AI (XAI) techniques such as Grad-CAM, SHAP, and LIME, the model enhances transparency and trust in AI-powered environmental monitoring and precision agriculture.

## 🚀 Features

- **Drone-based palm tree detection**
- **Deep Learning with Faster R-CNN + ResNet-50**
- **Transfer Learning using COCO-pretrained weights**
- **Explainability via Grad-CAM, SHAP, LIME**
- **Responsible AI principles including fairness and adversarial robustness**
- **Yield estimation from tree count**
- **Evaluation with IoU, mAP, Precision, Recall, F1 Score**

## 🧠 Technologies Used

- Python
- PyTorch
- torchvision
- COCO-style dataset
- Grad-CAM, SHAP, LIME for interpretability
- Google Colab (for training and visualization)

## 🗂 Dataset

- Aerial imagery of palm plantations
- COCO-format annotations ([x, y, width, height] -> converted to [xmin, ymin, xmax, ymax])
- Custom PyTorch `PalmTreeDataset` class for loading and preprocessing

## 📊 Model Architecture

- Backbone: ResNet-50 with Feature Pyramid Network (FPN)
- Detection Head: Faster R-CNN
- Custom classifier head: Binary classification (background, palm_tree)

## 📈 Evaluation Metrics

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 85%     |
| Precision     | 80%     |
| Recall        | 90%     |
| F1 Score      | 85%     |
| Train Accuracy| 99.21%  |
| Test Accuracy | 94.98%  |

## 📷 Visualization

- **Grad-CAM** heatmaps show model attention over tree crowns.
- Bounding boxes drawn over detected palm trees.
- Prediction confidence visualized over IoU and score graphs.

## ✅ Responsible AI Practices

- ✅ Transparency with visual explanations
- ✅ Robustness via adversarial training
- ✅ Fairness by training across diverse environments
- ✅ Scalable for real-world agricultural deployment

## 📌 Future Work

- Real-time inference on edge devices
- Integration of multispectral/thermal drone sensors
- Health prediction and disease monitoring
- Integration of LLM-based XAI tools for improved explainability

## 📁 Repository Structure

