---
title: 'Leetcode (ML)'
date: 2024-10-26 08:38:03 +00:00
tags: [maths, coding, ml, leetcode]
toc: false
---
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/contrib/auto-render.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false}
      ]
    });
  });
</script>

A friend found this nice website <a href="https://www.deep-ml.com/">deep-ml/</a> which attempts to re-create the "Leetcode experience" but for Machine Learning / Data Science. I'll try to solve some of these / write some refreshers on the topics.



# Problems

1. [Exercice 1](#matrix-times-vector)
2. [Exercice 2](#transpose-of-a-matrix)
3. [Exercice 3](#covariance-matrice)


## Matrix times a Vector

Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector

<details>
 <summary><strong>💡 Solution</strong></summary>

```python

def matrix_dot_vector(a:list[list[int|float]], b:list[int|float])-> list[int|float]:
    # Check if the number of columns in the matrix is equal to the number of rows in the vector
    if len(a[0]) != len(b):
        return -1
    dot_prod = []
    for row in a:
        s = 0
        for c_row, b_i in zip(row, b):
            s+= c_row * b_i
        dot_prod.append(s)
    return dot_prod

```

If we assume that $a$ and $b$ are NumPy arrays, we have a dedicated operator for this : 

```python

# Matrix a and vector b
dot_prod = a @ b

# Equivalent to
dot_prod = [row @ b for row in a]
```

<br><br><strong>Learning points ? </strong><br>

- The @ operator

</details>

<hr>

## Transpose of a Matrix

Write a Python function that computes the transpose of a given matrix.

<details>
 <summary><strong>💡 Solution</strong></summary>


```python

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    b = []
    for i in range(len(a[0])):
        new_row = []
        for j in range(len(a)):
            new_row.append(a[j][i])
        b.append(new_row)
    return b
```

Actually there is a neat pythonic way to do this using the `zip` function : 

```python

transposed = list(zip(*a))
```

`*a` unpacks every lists inside the main list (the rows of the matrix), `zip` as an iterator will then yield each index of the unpacked rows (tuple of the columns)...


If we assume that $a$ is a NumPy array :  

```python

# Matrix a and vector b
transposed = a.T
```

<br><br><strong>Learning points ? </strong><br>

- Pretty clever use of `zip` and `*` to transpose a matrix


</details>

<hr>

## Covariance Matrice

Write a Python function that calculates the covariance matrix from a list of vectors. Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.

<details>
 <summary><strong>💡 Solution</strong></summary>


A bit of a mindfuck because the inputs are not in the usual ML format. Here we consider a list of $F$ features, each having $N$ samples. <br> For instance : 

$$
M = 
\begin{bmatrix}
\text{height}_1 & \text{height}_2 & \dots & \text{height}_N \\
\text{age}_1 & \text{age}_2 & \dots & \text{age}_N \\
\text{weight}_1 & \text{weight}_2 & \dots & \text{weight}_N \\
\end{bmatrix}
$$
<br>

In that case, the covariance matrix can be *estimated* as $\frac{1}{N - 1} MM^{T}$, where $M$ is a centered data matrix

```python

import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	vectors = np.array(vectors)
	# Number of samples
	N = vectors.shape[1]
    # Mean of each features
	E = np.mean(vectors, axis=1)
    # Center each feature vector
	for i, mean in enumerate(E):
		vectors[i] = vectors[i] - mean
    # Final estimate
	covariance_matrix = (vectors @ vectors.T) / (N - 1)
	return covariance_matrix.tolist()
```

NumPy :<br>
```python
# rowvar=True because each row is a feature, usually in ML each column is a feature and each row a sample..
np.cov(vectors, rowvar=true)  
```
<br>

<br><br><strong>Learning points ? </strong><br>

- Why $\frac{1}{N - 1}$ ? Because we're estimating the covariance matrix from a sample, not the entire population (unbiased estimator). Otherwise, we would use $\frac{1}{N} MM^{T}$. <br>
- Since we have $F$ features, the covariance matrix is $F \times F$, where each entry $CoV[i, j]$ is the covariance between the $i^{th}$ and $j^{th}$ features. The diagonal will contain the variance of each feature (covariance with itself...).<br>
- The variance tells us how much a feature varies around its mean, the covariance tells us how much two features vary together (covariance positive if they vary together, negative if they vary in opposite directions).<br>
- The covariance matrix is symmetric (because the covariance between $i$ and $j$ is the same as the covariance between $j$ and $i$).<br>
- The covariance matrix is positive semi-definite by construction of the inner product $XX^{T}$ (all its eigenvalues are non-negative) <br>
- Used plenty in PCA (and other dimensionality reduction techniques), in finance (assets correlation / portfolio optimization) etc.


</details>

<hr>




