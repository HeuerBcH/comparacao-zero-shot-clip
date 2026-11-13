# comparacao-zero-shot-clip

Estudo comparativo de classificação zero-shot com CLIP ViT-B/32 sob mudança de domínio: prompt único (H0) contra Prompt Ensembling com 8 templates (H1), em 30 simulações pareadas sobre o CIFAR-100 com degradações sintéticas.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `experimento_clip.ipynb` | Notebook executado com todas as saídas |
| `zeroshot/` | Módulos: dados e shifts, modelo CLIP, experimento, estatística e gráficos |
| `resultados/simulacoes_30_rodadas.csv` | Métricas brutas por rodada e cenário |
| `resultados/resumo_estatistico.json` | Estatísticas descritivas, Shapiro-Wilk, teste pareado e d de Cohen |
| `figuras/` | Figuras em 300 DPI |
| `relatorio_ieee.tex` / `relatorio_ieee.pdf` | Artigo em formato IEEE Conference (fonte e PDF compilado) |
| `apresentacao_slides.md` | Roteiro de 7 slides |

## Reprodução

```bash
pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 experimento_clip.ipynb
```

O CIFAR-100 é baixado em `data/` na primeira execução. Em CPU a execução completa leva cerca de 15 minutos; com GPU CUDA disponível o dispositivo é detectado automaticamente. Acurácia e F1 são determinísticos; apenas os tempos variam entre execuções.

O PDF do artigo (`relatorio_ieee.pdf`) já está incluído, compilado com Tectonic (motor XeTeX). Para recompilar, envie `relatorio_ieee.tex` e a pasta `figuras/` ao Overleaf; o preâmbulo detecta o motor e funciona tanto com pdfLaTeX quanto com XeLaTeX.
