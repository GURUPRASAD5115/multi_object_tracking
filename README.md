# 🎯 Multi-Object Tracking System using YOLOv8 + DeepSORT

## 📌 Overview

This project implements a **Multi-Object Tracking (MOT) system** that detects and tracks multiple objects (specifically persons) in a video. Each detected object is assigned a **unique ID**, which is maintained across frames using a tracking algorithm.

The system is designed to handle real-world challenges such as **object motion, occlusion, and overlapping objects**.

---

## 🚀 Features

* 🔍 Real-time **Object Detection** using YOLOv8
* 🔁 **Multi-Object Tracking** using DeepSORT
* 🆔 Persistent **Unique ID Assignment**
* 🎯 Handles occlusion and motion effectively
* 📊 Displays:

  * Current object count (frame-based)
  * Total unique IDs (video-based)
* 📈 **Trajectory visualization** (movement paths)
* 🎥 Annotated output video generation

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries Used:**

  * OpenCV
  * NumPy
  * Ultralytics YOLOv8
  * DeepSORT (deep-sort-realtime)

---

## 📂 Project Structure

```
project/
│
├── main.py  
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository (or download zip)

```
git clone <your-repo-link>
cd project
```

### 2. Install Dependencies

```
pip install ultralytics opencv-python numpy deep-sort-realtime
```

---

## ▶️ How to Run

1. Place your input video as:

```
input.mp4
```

2. Run the program:

```
python main.py
```

3. Output will be saved as:

```
output.mp4
```

---

## 📊 Output Description

The output video contains:

* Bounding boxes around detected persons
* Unique tracking IDs (e.g., ID: 1, ID: 2)
* Trajectory paths of movement
* Current count of visible objects
* Total unique IDs detected

---

## 🧠 Methodology

### 1. Object Detection

* YOLOv8 is used to detect persons in each video frame.

### 2. Object Tracking

* DeepSORT tracks detected objects across frames using:

  * Motion (Kalman Filter)
  * Appearance features

### 3. ID Assignment

* Each object is assigned a unique ID that persists over time.

---

## ⚠️ Assumptions

* The input video contains **visible human subjects**
* Detection is limited to **person class only**
* Moderate lighting and resolution improve accuracy

---

## 🚧 Limitations

* ID switching may occur during heavy occlusion
* Fast motion may reduce tracking accuracy
* Counting is frame-based (not real-world entry/exit counting)

---

## ⭐ Future Improvements

* Entry/Exit based counting system
* Heatmap visualization
* Speed estimation of objects
* Multi-class tracking (vehicles, animals, etc.)
* Improved re-identification models

---

## 🎥 Demo

A short demo video (3–5 minutes) explains:

* Problem statement
* Approach used
* Live working output

---

## 📄 Conclusion

This project successfully demonstrates a **robust multi-object tracking system** using YOLOv8 and DeepSORT, capable of maintaining identity consistency across frames while handling real-world challenges.

---

## 👤 Author

* Rama Guru Prasad

---
