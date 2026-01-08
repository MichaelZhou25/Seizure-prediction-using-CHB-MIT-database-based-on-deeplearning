# Seizure Prediction using CHB-MIT Database based on Deep Learning

## Project Overview

This project implements a deep learning-based seizure prediction system using the CHB-MIT Scalp EEG Database. The system processes EEG signals to predict epileptic seizures before they occur by distinguishing between preictal (pre-seizure) and interictal (between seizures) states.

### Key Features

- **Data Processing Pipeline**: Automated extraction and preprocessing of EEG data from CHB-MIT database
- **Deep Learning Model**: Custom neural network combining:
  - Frequency and time domain feature extraction
  - Temporal Graph Convolutional Networks (TGCN)
  - Dynamic adjacency matrix learning for channel relationships
  - Bidirectional GRU for temporal modeling
- **Signal Processing**: Bandpass filtering, notch filtering, and Short-Time Fourier Transform (STFT)
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation

## Project Structure

```
.
├── preprocessing/          # Data preprocessing scripts
│   ├── useful_data_info_generating.py
│   ├── useful_data_extraction.py
│   ├── data_correction.ipynb
│   └── datachecking.ipynb
├── model/                  # Deep learning model
│   ├── model.py
│   ├── Basic_model_structure.png
│   └── model_flowchart.png
├── data/                   # Processed data (with overlap)
│   ├── useful_data_info.json
│   ├── patient_data_statistics.csv
│   ├── preictal/          # Preictal data segments
│   └── interictal/        # Interictal data segments
├── data_no_overlap/       # Processed data (without overlap)
│   ├── patient_data_statistics.csv
│   ├── preictal/
│   └── interictal/
└── README.md              # This file
```

## File Descriptions

### Preprocessing Directory (`preprocessing/`)

#### 1. `useful_data_info_generating.py`
**Purpose**: Generates valid data information for preictal and interictal periods from CHB-MIT database.

**Key Functions**:
- `parse_time_to_seconds()`: Converts time strings to seconds
- `summarize_edf_files()`: Parses EDF file metadata and calculates cumulative time
- `extract_seizures_with_cumulative_time()`: Extracts seizure events with timestamps
- `select_valid_seizures()`: Filters seizures with insufficient intervals (< 40 minutes)
- `generate_preictal_segments()`: Generates preictal segments (35 minutes before seizure, ending 5 minutes before)
- `extract_preictals_and_interictals()`: Extracts both preictal and interictal time segments

**Output**: `data/useful_data_info.json` - Contains time segments for each patient

**Configuration**:
- Preictal duration: 35 minutes
- Preictal end offset: 5 minutes before seizure
- Minimum seizure interval: 40 minutes
- Excluded time around seizures: 240 minutes

#### 2. `useful_data_extraction.py`
**Purpose**: Extracts and processes EEG data based on the information from `useful_data_info.json`.

**Key Functions**:
- `read_edf_segment()`: Reads specific time segments from EDF files
- `process_segments()`: Processes preictal/interictal segments for a patient

**Features**:
- Reads 22 target EEG channels
- Splits data into 5-second windows
- Applies overlapping or non-overlapping sampling based on class balance
- Saves processed data as HDF5 files

**Output**: 
- HDF5 files in `data/preictal/` and `data/interictal/`
- `patient_data_statistics.csv` - Statistics for each patient

**Parameters**:
- Sample length: 5 seconds
- Number of channels: 22
- Sampling overlap: Adjustable based on class imbalance

#### 3. `data_correction.ipynb` and `datachecking.ipynb`
**Purpose**: Jupyter notebooks for manual data validation and correction.

### Model Directory (`model/`)

#### 1. `model.py`
**Purpose**: Main deep learning model implementation with training and evaluation pipeline.

**Model Architecture Components**:

1. **FrequencyBranch**: Processes frequency domain features
   - Input: STFT log-magnitude spectrogram (128 frequency bins)
   - Local 1D convolutions for frequency feature extraction
   - Output: 64-dimensional frequency features per channel

2. **TimeBranch**: Processes time domain features
   - Input: Raw EEG signal (256 time points)
   - Temporal Convolutional Network (TCN) with dilated convolutions
   - Output: 64-dimensional temporal features per channel

3. **AdjacencyMatrixLearning**: Learns dynamic channel relationships
   - Generates adaptive adjacency matrix from frequency and time features
   - Outputs: (22×22) adjacency matrix and channel weights
   - Uses LayerNorm and Dropout for regularization

4. **TemporalGCN**: Graph convolutional network with temporal modeling
   - Multi-layer graph convolutions using learned adjacency matrix
   - Bidirectional GRU for temporal sequence modeling
   - Channel-wise attention pooling

5. **MainModel**: Complete end-to-end model
   - Integrates all components
   - Binary classification (preictal vs interictal)
   - Cross-entropy loss with Adam optimizer

**Signal Processing Functions**:
- `bandpass_filter()`: 4th-order Butterworth bandpass filter
- `notch_filter()`: Removes power line noise (60Hz, 120Hz)
- `remove_dc_component()`: Removes DC offset
- `apply_stft_to_data()`: GPU-accelerated STFT transformation

**Training Features**:
- 5-fold stratified cross-validation
- Early stopping (patience = 5 epochs)
- Learning rate: 0.001
- Batch size: 64
- Weight decay: 1e-4
- Training/validation loss and accuracy tracking
- Confusion matrix visualization

**Evaluation Metrics**:
- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-Score
- False Positive Rate per hour (FPR/h)

**Output**:
- Training logs in CSV format
- Loss and accuracy curves
- Confusion matrices
- Best model per fold

#### 2. `Basic_model_structure.png` and `model_flowchart.png`
Visual diagrams showing the model architecture and data flow.

### Data Directories

#### `data/` and `data_no_overlap/`
Two versions of processed data:
- `data/`: With overlapping windows (for data augmentation)
- `data_no_overlap/`: Non-overlapping windows (for testing)

**Contents**:
- `useful_data_info.json`: Time segment metadata for all patients
- `patient_data_statistics.csv`: Statistics summary
- `preictal/`: Preictal EEG segments in HDF5 format
- `interictal/`: Interictal EEG segments in HDF5 format

## Usage

### Prerequisites

```bash
pip install numpy scipy torch h5py pyedflib pandas scikit-learn matplotlib seaborn tqdm portion
```

### Step 1: Generate Data Information

```bash
cd preprocessing
python useful_data_info_generating.py
```

This will:
- Read CHB-MIT summary files
- Extract seizure timings
- Generate preictal and interictal time segments
- Save to `data/useful_data_info.json`

### Step 2: Extract and Process EEG Data

```bash
python useful_data_extraction.py
```

This will:
- Read EDF files based on time segments
- Extract 22 EEG channels
- Split into 5-second windows
- Save as HDF5 files with statistics

### Step 3: Train and Evaluate Model

```bash
cd ../model
python model.py
```

This will:
- Load processed data
- Apply signal preprocessing (filtering, STFT)
- Train model with 5-fold cross-validation
- Generate evaluation metrics and visualizations
- Save training logs

## Configuration

### Data Paths (modify in each script)
```python
# preprocessing/useful_data_info_generating.py
DATA_DIR = Path(r'path/to/CHB-MIT')
RESULT_DIR = Path(r'path/to/output/useful_data_info.json')

# preprocessing/useful_data_extraction.py
DATA_ROOT = Path(r'path/to/CHB-MIT')
SAVE_DIR = r"path/to/output"
USEFUL_DATA_INFO_ROOT = Path(r'path/to/useful_data_info.json')

# model/model.py
DATA_DIR = Path("path/to/processed/data")
PATIENT_ID = 5  # Patient to train on
```

### Model Hyperparameters (in `model/model.py`)
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PATIENCE = 5  # Early stopping patience
FS = 256  # Sampling rate
```

### Signal Processing Parameters
```python
N_FFT = 256  # FFT window size
HOP_LENGTH = 128  # STFT hop length
WIN_LENGTH = 256  # Window length
```

## Target EEG Channels (22 channels)

```
C3-P3, C4-P4, CZ-PZ, F3-C3, F4-C4, F7-T7, F8-T8, 
FP1-F3, FP1-F7, FP2-F4, FP2-F8, FT10-T8, FT9-FT10, 
FZ-CZ, P3-O1, P4-O2, P7-O1, P7-T7, P8-O2, T7-FT9, 
T7-P7, T8-P8
```

## Dataset

This project uses the **CHB-MIT Scalp EEG Database**:
- 24 patients (excluding patients 12, 13, 15, 24)
- Continuous EEG recordings
- Multiple seizure events per patient
- 256 Hz sampling rate
- Available at: https://physionet.org/content/chbmit/

## Model Performance

The model is evaluated using:
- 5-fold stratified cross-validation
- Balanced preictal/interictal samples
- Per-patient training (patient-specific models)

Metrics tracked:
- Validation accuracy
- Sensitivity (seizure detection rate)
- Specificity (avoiding false alarms)
- False positive rate per hour

## Notes

- The model processes each patient separately for patient-specific prediction
- Data is balanced by adjusting sampling overlap between classes
- GPU acceleration is used when available (CUDA)
- Training logs and statistics are saved for each fold
- Excluded patients (12, 13, 15, 24) have insufficient data

## Citation

If you use this code, please cite the CHB-MIT database:
```
Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components 
of a new research resource for complex physiologic signals." Circulation 
[Online]. 101.23 (2000): pp. e215–e220.
```

## License

Please refer to the repository license file.

## Author

MichaelZhou25

## References

- CHB-MIT Scalp EEG Database: https://physionet.org/content/chbmit/
- Related to Professor Chen's research group (陈教授组)