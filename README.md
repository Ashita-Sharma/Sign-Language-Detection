# 🤟 ASL Hand Gesture Recognition

A real-time American Sign Language (ASL) alphabet recognizer using computer vision and machine learning. Detects hand landmarks with MediaPipe and classifies gestures using a Random Forest classifier.

## ✨ Features

- **Real-time Detection**: Recognizes ASL letters A-Z through webcam
- **Hand Landmark Tracking**: Uses MediaPipe for accurate 3D hand pose estimation
- **Machine Learning Classification**: Random Forest model with high accuracy
- **Visual Feedback**: Displays predicted letter and hand landmarks on screen
- **Custom Dataset Creation**: Built-in tools to collect and process your own training data
- **Lightweight**: Fast inference suitable for real-time applications

## 🎯 How It Works

1. **Data Collection**: Captures 300 images per ASL letter via webcam
2. **Feature Extraction**: MediaPipe extracts 21 hand landmarks (x, y, z coordinates = 63 features)
3. **Model Training**: Random Forest classifier learns from landmark patterns
4. **Real-time Prediction**: Detects hand, extracts landmarks, and predicts letter

## 🚀 Installation & Setup

### Prerequisites

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

**Requirements:**
- Python 3.7+
- Webcam
- opencv-python
- mediapipe
- scikit-learn
- numpy

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Ashita-Sharma/ASL-Hand-Gesture-Recognition.git
cd ASL-Hand-Gesture-Recognition
```

## 📋 Usage Pipeline

### Step 1: Collect Training Data

```bash
python image_collector.py
```

**What it does:**
- Creates a `data/` directory with 26 subdirectories (0-25 for A-Z)
- Collects 300 images per class
- Press **'Q'** to start capturing for each letter

**Tips:**
- Use different hand positions and angles
- Vary lighting conditions for robustness
- Keep hand clearly visible in frame

### Step 2: Process Data & Extract Features

```bash
python data_creator.py
```

**What it does:**
- Processes all collected images
- Extracts 21 hand landmarks (63 features: x, y, z per landmark)
- Saves processed data to `data.pickle`
- Outputs: "Collected X samples"

### Step 3: Train the Model

```bash
python classifer_trainer.py
```

**What it does:**
- Loads processed landmark data
- Splits data: 80% training, 20% testing
- Trains Random Forest classifier
- Saves trained model to `model.p`
- Outputs accuracy: "X% of samples classified correctly"

### Step 4: Test Real-time Recognition

```bash
python classifier_tester.py
```

**What it does:**
- Opens webcam feed
- Detects hand landmarks in real-time
- Predicts ASL letter
- Displays prediction with bounding box
- Press **'Q'** to quit

## 🏗️ Project Structure

```
ASL-Hand-Gesture-Recognition/
├── image_collector.py         # Collect training images
├── data_creator.py           # Extract hand landmarks
├── classifer_trainer.py      # Train ML model
├── classifier_tester.py      # Real-time prediction
├── data.pickle               # Processed landmark data
├── model.p                   # Trained classifier
└── data/                     # Training images directory
    ├── 0/                    # Letter A images
    ├── 1/                    # Letter B images
    └── ...                   # Letters C-Z
```

## 🔧 Technical Details

### Hand Landmarks

MediaPipe detects **21 landmarks** per hand:
- Wrist (1 point)
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky (4 points)

Each landmark has **3 coordinates** (x, y, z) = **63 total features**

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**: 63 (21 landmarks × 3 coordinates)
- **Classes**: 26 (A-Z)
- **Train/Test Split**: 80/20
- **Stratified Sampling**: Ensures balanced class distribution

### Classification Pipeline

```
Camera Feed → BGR to RGB → MediaPipe Detection → 
Landmark Extraction (63 features) → Random Forest → Letter Prediction
```

## ⚙️ Customization

### Adjust Dataset Size

In `image_collector.py`:
```python
dataset_size = 300  # Change to collect more/fewer images per class
```

### Modify Number of Classes

To recognize fewer letters:
```python
number_of_classes = 10  # Only A-J
```

Update `labels_dict` in `classifier_tester.py`:
```python
labels_dict = {i: chr(65 + i) for i in range(10)}
```

### Change Model Parameters

In `classifer_trainer.py`:
```python
model = RandomForestClassifier(
    n_estimators=200,      # More trees = better accuracy
    max_depth=20,          # Control overfitting
    random_state=42
)
```

### Adjust Detection Confidence

In `classifier_tester.py`:
```python
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,  # Increase for stricter detection
    min_tracking_confidence=0.5
)
```

## 📊 Performance Tips

**To improve accuracy:**
1. Collect more diverse training data (different backgrounds, lighting)
2. Increase `dataset_size` to 500+ images per class
3. Use `n_estimators=200` or higher in Random Forest
4. Ensure consistent hand positioning during data collection
5. Clean data by removing unclear/incorrect images

**If detection is slow:**
1. Reduce camera resolution
2. Use fewer estimators in Random Forest
3. Skip frames: `if frame_count % 2 == 0:` process frame

## 🐛 Troubleshooting

**Camera not opening:**
```python
cap = cv2.VideoCapture(1)  # Try different camera indices (0, 1, 2)
```

**Low accuracy:**
- Collect more training data
- Ensure hand is well-lit during capture
- Check that all 26 classes have equal samples

**"No hand detected":**
- Improve lighting
- Lower `min_detection_confidence` to 0.3
- Move hand closer to camera

**Import errors:**
```bash
pip install --upgrade opencv-python mediapipe scikit-learn
```

## 📈 Expected Results

With proper data collection:
- **Training Accuracy**: 95-99%
- **Real-time FPS**: 20-30 FPS
- **Inference Time**: ~30-50ms per frame

## 🎯 Future Enhancements

- [ ] Add dynamic gestures (words, not just letters)
- [ ] Multi-hand support (two-handed signs)
- [ ] Add numbers (0-9)
- [ ] Implement LSTM for gesture sequences
- [ ] Create mobile app version
- [ ] Add voice output for predicted letters
- [ ] Sentence formation and word suggestions
- [ ] Support for other sign languages (BSL, ISL, etc.)

## 📚 Learning Resources

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [ASL Alphabet Guide](https://www.lifeprint.com/asl101/fingerspelling/abc.htm)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better model architectures (CNN, LSTM)
- Data augmentation techniques
- GUI interface
- Performance optimizations

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **OpenCV** for computer vision tools
- **scikit-learn** for machine learning utilities

## 📧 Contact

**Ashita Sharma**
- GitHub: [@Ashita-Sharma](https://github.com/Ashita-Sharma)

---

⭐ **Star this repo if you find it useful!** ⭐

**Happy Signing! 🤟✨**
