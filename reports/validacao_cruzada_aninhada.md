# Validação cruzada aninhada vs. não aninhada (AUC-ROC)

Gerado em: 2026-10-11T22:53:47+00:00

| Modelo | Não aninhada (média ± desvio) | Aninhada (média ± desvio) | Δ média |
| --- | --- | --- | --- |
| Random Forest | 0.9102 ± 0.0187 | 0.9106 ± 0.0176 | +0.0004 |
| XGBoost | 0.9090 ± 0.0164 | 0.9097 ± 0.0178 | +0.0007 |

A versão aninhada é a estimativa honesta de desempenho (sem o
otimismo da busca de hiperparâmetros não aninhada); a diferença
para a versão não aninhada explica, ao menos em parte, a queda
observada entre a AUC-ROC de validação cruzada (~0.91 na versão
original do artigo) e a AUC-ROC da divisão única de teste
(~0.86-0.87).
