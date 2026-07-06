# AgriScan

AgriScan is a Python-based Streamlit application for crop disease detection from leaf images. It runs a TensorFlow model to predict likely plant diseases and provides practical treatment guidance for farmers.

## Features

- Leaf image upload workflow (`.jpg`, `.jpeg`, `.png`)
- AI-powered disease prediction using a trained TensorFlow model
- Top-3 prediction confidence breakdown
- Built-in treatment/advice mapping for supported disease classes
- Farmer-focused interface with image capture tips and confidence warnings

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **ML Inference:** TensorFlow (CPU)
- **Image Processing:** Pillow
- **Numerical Processing:** NumPy

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/hackerbotfz/AgriScan.git
cd AgriScan
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows (PowerShell)
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit app from the repository root:

```bash
streamlit run AgriScan.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`), upload a clear leaf image, and review diagnosis confidence plus treatment guidance.

## Project Structure

```text
AgriScan/
├── AgriScan.py          # Main Streamlit application and inference flow
├── agriscan_model.h5    # Trained TensorFlow model used for predictions
├── class_indices.json   # Class index mapping used to decode model outputs
└── requirements.txt     # Python dependencies
```

## Configuration

AgriScan currently does **not** require environment variables.

Runtime paths and inference settings are configured in `AgriScan.py` via constants:

- `MODEL_PATH`
- `CLASS_INDEX_PATH`
- `IMG_SIZE`
- `TOP_K`

If you move model assets or want to change inference behavior, update these constants accordingly.

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make focused, well-tested changes
4. Open a pull request with a clear description

For larger changes, consider opening an issue first to discuss scope and approach.

## License

No license file is currently present in this repository. If you are the maintainer, consider adding a `LICENSE` file to clarify usage and contribution terms.

## Contact / Support

For questions or support:

- Open a GitHub issue in this repository
- Contact the maintainer: **@hackerbotfz**
