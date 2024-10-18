---
title: "Random Variables are NOT Random and NOT Variables"
date: 2024-10-14 08:00:00 +00:00
tags: [maths, probability]
toc: false
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Probabilities can be confusing and I'm still wrapping my head around some (simple) concepts. So...

# Random Variables are NOT Random

The term "random variable" is actually a misnomer, it is formally defined as a deterministic function that takes as input any element from the sample space and returns a number from the set of possible results of the experiment (the sample space $\Omega$ is technically part of the probability space $(\Omega, \mathcal{F}, P)$ 🤓). <br> What's random is the underlying process that generates the outcomes (whatever that means, let's not get philosophical, like rolling a dice).<br>
Another way of seeing it, is to see a random variable as a collection of events, in a way we're "slicing" our sample space into different outcomes. For example, when we say $\{X = 3\}$, we're actually looking at the set $\{\omega \in \Omega : X(\omega) = 3\}$...

# Random Variables are NOT Variables

They're not variables in the traditional sense, like our unknown $x$ in algebra which we try to solve for. They're deterministic functions that map atoms from the sample space to possible outcomes. However, in probability theory, we view them as the result of a random experiment which motivates the specific notations (like $X = a$ or $X \in A$). For all intents and purposes, they're manipulated as variables, and we can do computations with them.

## Dice Roll

Let's say we consider 2 fair four-sided dices (it's for the sake of the argument, it's like a nerd dice for dungeons and dragons). <br>
The sample space $\Omega$ is the set of all possible outcomes, which is the cartesian product of the two dices, i.e $$\Omega = \{1,2,3,4\} \times \{1,2,3,4\}$$<br>

$$
\begin{equation}
\Omega = \left\{
\begin{array}{cccc}
(1,1) & (1,2) & (1,3) & (1,4) \\
(2,1) & (2,2) & (2,3) & (2,4) \\
(3,1) & (3,2) & (3,3) & (3,4) \\
(4,1) & (4,2) & (4,3) & (4,4)
\end{array}
\right\}
\end{equation}
$$

What we can already say is that, since the dices are fair, each element $$\omega \in \Omega$$ has a probability of $$\frac{1}{16}$$ of happening. <br>

If we consider the random variable $X$ representing the sum of the two dices, i.e $X = X_1 + X_2$ where $X_1$ and $X_2$ are the outcomes of the first and second dice respectively. The possible values of $X$ are $\{2,3,4,5,6,7,8\}$ and we have $X: \Omega \mapsto \{2,3,4,5,6,7,8\}$.<br>

Something we often want to do is to compute the expected value of a random variable (the average in some sense). We can do it in a few ways, either by considering every single element of the sample space weighted by their respective probabilities of happening : 
$$\mathbb{E}[X] = \sum_{\omega \in \Omega} X(\{\omega\}) \cdot P(\{\omega\}) = 2 \times \frac{1}{16} + 3 \times \frac{1}{16} + ... + 8 \times  \frac{1}{16} = 5$$
(For the first $\omega = (1,1)$, $X(\{(1, 1)\}) = 1 + 1 = 2$ and $P(\{\omega\}) = \frac{1}{16}$  etc.)<br>

But since we treat $X$ as a variable, we can also write : 
$$\mathbb{E}[X] = \sum_{\omega \in \Omega} X(\{\omega\}) \cdot P(\{\omega\}) = \sum_{\omega_{1} \in \Omega_{1}} X_1(\{\omega_{1}\}) \cdot P(\{\omega_{1}\}) + \sum_{\omega_{2} \in \Omega_{2}} X_1(\{\omega_{2}\}) \cdot P(\{\omega_{2}\}) = \mathbb{E}[X_1] + \mathbb{E}[X_2] = 2.5 + 2.5 = 5$$

(Not being super rigorous here but we have $\Omega = \Omega_1 \times \Omega_2$ i.e $\omega = (\omega_1, \omega_2)$ with $\Omega_1 = \Omega_2 = \{1,2,3,4\}$ the respective sample spaces of each dice)<br>

The average value of 2 rolls is the average value of the first roll + the second !

Since we're looking at the sum of the two dices, we can also group every $\omega \in \Omega$ into buckets called events by their sum, i.e $\{2\} = \{(1,1)\}$, $\{3\} = \{(1,2), (2,1)\}$. Each bucket has its own probability of happening, for instance there is only one way to get a sum of 2, so $P(\{2\}) = \frac{1}{16}$, but there are 2 ways to get a sum of 3, so $P(\{3\}) = \frac{2}{16}$ etc. <br>

The expected value can then be computed as:
$$\mathbb{E}[X] = \sum_{x \in \{2, 3, 4, 5, 6, 7, 8\}} x \cdot P(X = x) = 2 \times \frac{1}{16} + 3 \times \frac{2}{16} + ... + 8 \times \frac{1}{16} = 5$$

In a way, the first formula focuses on the start of the process (all the individual outcomes) while the second one focuses on the end (the sum of the dices). Usually we work with the second one because it's easier to compute but the first one can come in handy (for instance in the proof of <a href="https://en.wikipedia.org/wiki/Markov%27s_inequality">Markov's inequality</a>).

## Coin Flip

We now want to find out the average number of heads in $n$ coin flips. Here the sample space would be $\Omega = \{H, T\}^n$ ($\|\Omega\| = 2^{n}$ !).

Let $X$ be the random variable representing the number of heads in $n$ coin flips, possible values are $\{0, 1, 2, ..., n\}$ (never getting any heads to getting all heads).<br>

If we think a bit, we have $2^{n}$ possible sequences of heads and tails, so getting $x$ number of heads is just counting how many sequences of exactly $x$ heads we can choose from $n$ coin flips, i.e $\binom{n}{x}$ (the number of ways to choose $x$ elements from a set of $n$ elements), so $P(X = x) = \binom{n}{x} \cdot \left(\frac{1}{2}\right)^{n}$.<br>

The expected value of $X$ is then :
$$\mathbb{E}[X] = \sum_{x = 0}^{n} x \cdot P(X = x) = \sum_{x = 0}^{n} x \cdot \binom{n}{x} \cdot \left(\frac{1}{2}\right)^{n}$$

This sum is not trivial at all to compute, I'll have to think about it but long story short the answer is $\frac{n}{2}$.<br>

Another way to look at the problem is to think of our number of heads as the result of each individual coin flip, i.e $X = X_1 + X_2 + ... + X_n$ where $X_i = 1$ if the $i^{th}$ coin flip is a head and $0$ otherwise. The expected value of $X$ is then the sum of the expected value of each $X_i$ which is $\mathbb{E}[X_i] = 0 \times \frac{1}{2} + 1 \ times \frac{1}{2} $ (since the probability of getting a head is $\frac{1}{2}$).<br> 

So $\mathbb{E}[X] = \mathbb{E}[X_1] + \mathbb{E}[X_2] + ... + \mathbb{E}[X_n] = n \times \frac{1}{2} = \frac{n}{2}$.

Ok thanks