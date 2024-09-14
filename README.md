# Sentiment Analysis using Convolutional Neural Network (CNN)

This application performs sentiment analysis using a CNN (Convolutional Neural Network) model. The dataset is first labeled and processed, and then the data is classified into positive and negative sentiments. This project uses word embeddings (cc.id.300.vec) and involves multiple stages, including data labeling and processing.

## Files and Directories

- **komentar.csv**: The main dataset file containing comments that will be analyzed.
- **normalization.xlsx**: Excel file containing normalization rules for text preprocessing.
- **stopwords.txt**: Text file containing stopwords to be filtered out during preprocessing.
- **label.py**: Script that processes `komentar.csv` and labels the data into positive and negative sentiments. Outputs two files: `data.csv` and `data.json`.
- **processing.py**: Processes the labeled data from `data.csv` and performs further analysis, including training the CNN model and visualizing the results.
- **cc.id.300.vec**: Pre-trained word embedding for the Indonesian language from the fastText project, used to represent words in a dense vector format.
- **requirement.txt**: Contains all required Python dependencies.

## Requirements

- Python version >= 3.10.3
- Dependencies specified in `requirement.txt`
- Pre-trained word embeddings from fastText: [cc.id.300.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz)

## Setup Instructions

Follow the steps below to set up the environment and run the application:

1. **Create a virtual environment**:

   ```bash
   python -m venv env

   ```

2. **Create a Python Virtual Environment**:

#### On Windows

```bash
python -m venv env
. env/Scripts/activate
```

#### On Mac/linux

```bash
python -m venv env
. env/bin/activate
```

3. **Install Dependencies**

Install the necessary dependencies listed in requirement.txt by running the following command:

```bash
pip install -r requirement.txt
```

4. **Download word embeddings:**

Download the pre-trained word embeddings from the fastText repository: [cc.id.300.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz). Ensure the file is in the correct directory.

## Usage

### Step 1: Labeling the Data

The first step is to label the comments in the komentar.csv file into positive and negative categories.

```bash
python label.py
```

This will generate two files:

- `data.csv`: Labeled data in CSV format.
- `data.json`: Labeled data in JSON format.

### Step 2: Processing and Visualizing the Data

Next, process the labeled data and visualize the sentiment analysis.

```bash
python processing.py
```

## Directory Structure

```bash
sentiment-cnn/
│
├── komentar.csv              # Input comments data
├── normalization.xlsx        # Text normalization rules
├── stopwords.txt             # Stopwords for filtering
├── label.py                  # Script for labeling sentiment data
├── processing.py             # Script for processing labeled data
├── cc.id.300.vec             # Pre-trained word embeddings
├── requirement.txt           # Python dependencies
└── README.md                 # Project documentation

```

## Dependencies

The following libraries are required for running the scripts:

- **Python 3.10.3 or higher**
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **textblob**: For sentiment analysis (used in `label.py`).
- **deep_translator**: For translating text (used in `label.py`).
- **scikit-learn**: For model evaluation and splitting datasets (used in `processing.py`).
- **gensim**: For working with word embeddings and Word2Vec (used in `processing.py`).
- **tensorflow**: For building and training the CNN model (used in `processing.py`).
- **matplotlib**: For plotting graphs (used in `processing.py`).
- **seaborn**: For creating advanced visualizations (used in `processing.py`).
- **wordcloud**: For generating word cloud visualizations (used in `processing.py`).
- **keras**: For defining and training the CNN model (used in `processing.py`).
- **openpyxl**: For reading Excel files (used in `processing.py`).

All dependencies are listed in the `requirement.txt` file and can be installed using pip:

```bash
pip install -r requirement.txt
```

## Acknowledgments

- Word embeddings from [fastText](https://fasttext.cc/).
- CNN architecture for sentiment analysis inspired by various academic works on natural language processing.

### License

This project is licensed under the [MIT License](https://github.com/kansaok/sentiment-cnn?tab=MIT-1-ov-file) - see the LICENSE file for details.
