---
title: "The Man Who *Didn't* Solve The Market"
date: 2024-08-28 08:38:03 +00:00
tags: [maths, coding, probability]
toc: false
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


# What are we doing here ?

Yeah, so recently I've been reading "The Man Who Solved The Market" by Gregory Zuckerman. Fun book about Jim Simons and his famously cracked Renaissance Technologies hedge fund. Unfortunately, it didn't make me better at trading lol

<figure style="text-align: center;">
  <img src="/assets/img/rekt.png" alt="HL" style="width: 60%; max-width: 500px;">
  <figcaption style="font-style: italic;">Use my <a href="https://app.hyperliquid.xyz/join/CYRIL9227">ref link</a>👽</figcaption>
</figure>

but it motivated me to learn more about some of the maths they used. Notably, Markov chains are mentioned a few times, first in the context of the IDA, the Institute for Defense Analyses, where Simons worked before founding Renaissance, he published a paper
Simons published a paper while working at the IDA in Princeton in 1964 on Markov Chains. So I thought I'd write a post about it.

>Simons and the code-breakers proposed a similar approach to predicting stock prices, relying on a sophisticated mathematical tool called a hidden Markov model. Just as a gambler might guess an opponent’s mood based on his or her decisions, an investor might deduce a market’s state from its price movements.  Simons’s paper was crude, even for the late 1960s. He and his colleagues made some naive assumptions, such as that trades could be made “under ideal conditions,” which included no trading costs, even though the model required heavy, daily trading. Still, the paper can be seen as something of a trailblazer.

# Markov Chains

Big assumption of markov models is that the probability of the next state depends only on the current state and not on the sequence of events that preceded it. This is called the Markov property.
Formally, if we note the states of the system as $$X_{0}, X_{1}, X_{2}, \ldots$$, the Markov property states that for all $$n \geq 0$$ we have:
$$P(X_{n+1} | X_{n}, X_{n-1}, \ldots, X_{0}) = P(X_{n+1}| X_{n})$$

A Markov chain is usually given by a graph :


<img src="/assets/img/mchain.png" alt="mchain">


where each node is a possible state and each edge represents the transition probability.

A markov chain is entirely described by 3 components. An initial probability distribution $$\pi$$, a transition probability matrix $$A$$ where each $$a_{ij}$$ representing the probability of moving from state
$$i$$ to state $$j$$ and a list of possible states $$q_{1} ... q_{n}$$
    


The transition matrix for this Markov chain is a square matrix where each element represents the probability of transitioning from one state to another. The rows correspond to the current state, and the columns correspond to the next state.

Given our 3 states $$HOT$$, $$COLD$$ and $$WARM$$, the transition matrix $$A$$ is then:


$$A = \begin{pmatrix}
P(\text{HOT} \rightarrow \text{HOT}) & P(\text{HOT} \rightarrow \text{COLD}) & P(\text{HOT} \rightarrow \text{WARM}) \\
P(\text{COLD} \rightarrow \text{HOT}) & P(\text{COLD} \rightarrow \text{COLD}) & P(\text{COLD} \rightarrow \text{WARM}) \\
P(\text{WARM} \rightarrow \text{HOT}) & P(\text{WARM} \rightarrow \text{COLD}) & P(\text{WARM} \rightarrow \text{WARM})
\end{pmatrix}$$

Substituting the values from the diagram:

$$A = \begin{pmatrix}
0.6 & 0.1 & 0.3 \\
0.1 & 0.8 & 0.1 \\
0.3 & 0.1 & 0.6
\end{pmatrix}$$

This matrix represents the probabilities of moving from one state to another. Each row sums to $$1$$, which is a requirement for a valid transition matrix in a Markov chain.


# Hidden Markov Chains

# Ressources

https://web.stanford.edu/~jurafsky/slp3/A.pdf
https://github.com/Bratet/Stock-Prediction-Using-Hidden-Markov-Chains/blob/main/src/utils/hmm.py