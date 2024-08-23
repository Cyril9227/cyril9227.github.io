---
layout: default
title: LLG
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

# Mes solutions aux problèmes du ["PDF LLG"](https://www.louislegrand.fr/wp-content/uploads/2022/01/EXOS-TERMINALE3-3-AVECDESSIN.pdf)

## Introduction

Page en léger, où le but est de résoudre les problèmes du PDF LLG, et de les poster ici. J'essaie quand c'est possible de donner une solution différente de celle donnée dans le corrigé officiel et de faire le lien avec d'autres notions ou problèmes. 

## Liste des problèmes

1. [Exercice 266](#exercice-266-tout-ce-qui-est-possible-finit-par-arriver)


## Exercice 266 (Tout ce qui est possible finit par arriver)

On lance un dé non pipé. On répète $n$ fois l'opération, les lancers successifs étant supposés indépendants. Quelle est la probabilité $p_n$ pour que l'on obtienne au moins un 6 ? Déterminez la limite de $(p_n)_{n \geq 1}$.

<details>
 <summary>Solution</summary>
On peut considérer l'événement contraire $\overline{A_{k}}$ : "ne pas obtenir 6 lors du k-ième lancer" dont la probabilité est $P(\overline{A_{k}}) = \dfrac{5}{6}$

N'obtenir aucun 6 lors des $n$ lancers est donc l'événement $\overline{A} = \cap_{k=1}^{n} \overline{A_{k}}$ dont la probabilité est, par indépendance des $\overline{A_{k}}$, $P(\overline{A}) = \prod_{k=1}^{n} P(\overline{A_{k}}) = \left(\dfrac{5}{6}\right)^n$ 

La probabilité de l'événement contraire $A$ : "obtenir au moins un 6 lors des $n$ lancers" est donc $p_n = 1 - P(\overline{A}) = 1 - \left(\dfrac{5}{6}\right)^n$, et la limite de $(p_n)_{n \geq 1}$ est donc 1 (car $\left(\dfrac{5}{6}\right)^n \to 0$).
</details>

### Ce que j'ai appris ? 

Et voilà, c'était pas bien compliqué. On retiendra que "au moins" = "complémentaire de aucun", passer au complémentaire doit être un réflexe.

