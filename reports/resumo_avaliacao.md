# Resumo da avaliação dos modelos

Gerado em: 2026-10-11T22:57:43+00:00

Consolida, num único documento, o que os demais relatórios desta
etapa (`comparacao_modelos.md`, `validacao_cruzada.md`, `comparacao_ajuste_hiperparametros.md`, `importancia_features.png`) mostram separadamente.

## Desempenho final (validação cruzada, hiperparâmetros ajustados)

| Modelo | AUC-ROC (média ± desvio) |
| --- | --- |
| Random Forest | 0.9102 ± 0.0187 |
| XGBoost | 0.9090 ± 0.0164 |

**Melhor modelo: Random Forest** (AUC-ROC média = 0.9102 ± 0.0187). Detalhamento por fold em `validacao_cruzada.md`.

## Efeito da busca de hiperparâmetros

| Modelo | Padrão (média ± desvio) | Ajustado (média ± desvio) | Δ média |
| --- | --- | --- | --- |
| Random Forest | 0.8153 ± 0.0406 | 0.9102 ± 0.0187 | +0.0949 |
| XGBoost | 0.8888 ± 0.0172 | 0.9090 ± 0.0164 | +0.0202 |

Detalhamento completo em `comparacao_ajuste_hiperparametros.md`.

## AUC-ROC de uma única divisão treino/teste (referência)

| Modelo | AUC-ROC |
| --- | --- |
| Random Forest | 0.9058 |
| XGBoost | 0.9102 |

Só como referência -- com o rótulo de risco raro nos dados reais
deste projeto (poucos município-anos de risco), uma única divisão
dá uma estimativa mais ruidosa que a validação cruzada acima (ver
seção anterior e `validacao_cruzada.md`). Detalhamento em `comparacao_modelos.md`.

## Features mais importantes

- **Random Forest**: ano (0.636), casos_lag_1 (0.144), populacao (0.106)
- **XGBoost**: ano (0.585), casos_lag_1 (0.124), populacao (0.120)

Gráfico completo em `importancia_features.png`; curvas ROC em `curvas_roc.png`.
