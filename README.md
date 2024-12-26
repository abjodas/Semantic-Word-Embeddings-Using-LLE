
# Semantic Word Embeddings using Locally Linear Embedding (LLE)

This repository contains a Jupyter Notebook that demonstrates the use of Locally Linear Embedding (LLE) for dimensionality reduction on semantic word embeddings. The goal is to visualize high-dimensional word embeddings in a lower-dimensional space while preserving local neighborhood relationships.



## Features

- Loading and preprocessing word embeddings
- Implementing Locally Linear Embedding (LLE)
- Visualizing high-dimensional embeddings in 2D or 3D
- Analyzing neighborhood preservation in reduced space



## Prerequisites

To run the notebook, ensure you have the following installed:

- Python 3.x <img src="https://brandslogos.com/wp-content/uploads/images/large/python-logo.png" alt="drawing" width="20" align="center"/>
- Jupyter Notebook or JupyterLab <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" alt="drawing" width="20" align="center"/>
- NumPy <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1200px-NumPy_logo_2020.svg.png" alt="drawing" width="20" align="center"/>
- Pandas <img src="https://seeklogo.com/images/P/pandas-icon-logo-BE10401BF1-seeklogo.com.png" alt="drawing" width="20" align="center"/>
- Matplotlib <img src="https://www.jumpingrivers.com/blog/customising-matplotlib/matplot_title_logo.png" alt="drawing" width="20" align="center"/>
- Scikit-learn <img src="https://quintagroup.com/cms/python/images/scikit-learn-logo.png/@@images/4a0dce0a-be5d-4d11-a913-f53f9e5abf16.png" alt="drawing" width="20" align="center"/>
- Other optional dependencies listed in the notebook

## Screenshots
![Unknown-10](https://github.com/user-attachments/assets/23ba7287-031e-4e2e-867d-902265149567)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/semantic-word-embeddings.git
```
Navigate to the directory:

```
cd semantic-word-embeddings
```

Launch the Jupyter Notebook:
```
jupyter notebook LLE_Semantic_Word_Embeddings.ipynb
```
## Usage

Open the notebook and execute the cells step by step to:

- Load a pre-trained word embedding model or your own embeddings
- Apply LLE for dimensionality reduction
```python
def apply_lle(embeddings, n_neighbors=5, n_components=2):
  lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
  reduced_embeddings = lle.fit_transform(embeddings)
  return reduced_embeddings
```
- Visualize and analyze the reduced embeddings
```python
def visualize_embeddings(reduced_embeddings, words):
  plt.figure(figsize = (10,8))
  for i, word in enumerate(words):
    plt.scatter(reduced_embeddings[i,0], reduced_embeddings[i,1], marker = 'o', color = 'blue')
    plt.text(reduced_embeddings[i,0], reduced_embeddings[i,1], word, fontsize = 12)
  plt.title('LLE semantic word embeddings')
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.grid(True)
  plt.show()
```

## Examples

**Visualization**

Generate 2D or 3D scatter plots of word embeddings to observe semantic relationships.

**Neighborhood Analysis**

Evaluate how well LLE preserves local neighborhoods in the reduced space.




## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements or additional features.



## License

This project is licensed under the MIT License.

[MIT](https://choosealicense.com/licenses/mit/)

