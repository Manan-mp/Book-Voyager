**Book Voyager**
A lightweight hybrid recommender that suggests books you might enjoy by combining

Content-based similarity (TF-IDF on title + author)

Collaborative filtering (SVD on user ratings)

A simple weighted hybrid of the two.

Everything lives in one Jupyter notebook (ai-project.ipynb).

Quick Start
1 . Clone / download this repo
text
git clone <your-repo-url>
cd book-recommender
2 . Install Python dependencies
The exact versions used in the notebook:

text
pip install numpy==1.24.4 kaggle scikit-surprise networkx matplotlib seaborn
3 . Get the data
Kaggle dataset: “goodbooks-10k”

text
# ①  Place your kaggle.json in the project root or run:
from google.colab import files; files.upload()   # in Colab

# ②  Then execute:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# ③  Download & unzip
kaggle datasets download -d zygmunt/goodbooks-10k
unzip goodbooks-10k.zip -d goodbooks
4 . Run the notebook
Open ai-project.ipynb (locally or in Google Colab) and run the cells top-to-bottom.
You’ll get:

Top-rated books preview

Content-based recommendations

SVD model training + error metrics (RMSE / MAE)

Hybrid recommender

A tiny CLI that asks for a user ID and a favourite title and prints the top suggestions.

Folder / file overview
File / folder	Purpose
ai-project.ipynb	All code — data load, modelling, evaluation, mini-CLI
goodbooks/	Unzipped CSVs from the Goodbooks-10k dataset
README.md	This guide
Model details
Content model – TfidfVectorizer + cosine similarity on “title + author” text.

Collaborative model – Surprise SVD (100 latent factors, 20 epochs).

Hybrid – 60% content score + 40% normalised SVD score.

Average evaluation on a 80/20 split: RMSE ≈ 0.84, MAE ≈ 0.66.

Customising
Tweak the blending weights in hybrid_recommend() to favour content vs. collaborative.

Increase n_factors, n_epochs or provide the full ratings matrix (remove the random 1 000-book sampling) for better accuracy.

Replace TF-IDF with embeddings (e.g. Sentence-BERT) for richer content similarity.

License
The code here is provided “as-is”, without warranty. The Goodbooks-10k dataset is distributed under its own license on Kaggle.
