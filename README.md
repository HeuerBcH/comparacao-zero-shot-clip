# comparacao-zero-shot-clip

Estudo comparativo de classificação zero-shot com **CLIP ViT-B/32** sob mudança de domínio: **prompt único (H0)** contra **Prompt Ensembling com 8 templates (H1)**, em **30 simulações pareadas** sobre o CIFAR-100 com degradações sintéticas.

Trabalho da disciplina de Inteligência Artificial da CESAR School.
**Autores:** Silvio Fittipaldi, Bernardo Heuer e Rodrigo Nunes.

---

## Atendimento aos requisitos da atividade

| Requisito | Onde foi atendido |
|---|---|
| 3 técnicas vistas em aula | **A.** Vision Transformer (ViT-B/32) · **B.** CLIP (alinhamento multimodal contrastivo) · **C.** Estratégia de inferência textual (engenharia de prompt) |
| 2 cenários: A + B + C contra A + B + C (alternativa) | **H0:** ViT + CLIP + prompt único `"a photo of a {classe}."` · **H1:** ViT + CLIP + Prompt Ensembling (média L2-normalizada de 8 templates) |
| 30 simulações e inferência H0 × H1 | Sementes 1 a 30, 500 imagens estratificadas por rodada, teste pareado com α = 0,05 (`zeroshot/stats.py`) |
| Código com caso de uso | `experimento_clip.ipynb` + pacote `zeroshot/` |
| Apresentação | `apresentacao_slides.md` (roteiro de 7 slides com falas e indicação dos gráficos) |
| Relatório IEEE coluna dupla com gráficos | `relatorio_ieee.tex` (classe `IEEEtran`, modo `conference`) e `relatorio_ieee.pdf` compilado, com 4 figuras e 2 tabelas |

## Hipóteses

Seja Δᵢ = Acc(H1) − Acc(H0) a diferença de acurácia Top-1 na rodada *i*:

- **H0 (nula):** μ_Δ = 0. O Prompt Ensembling não altera a acurácia zero-shot sob mudança de domínio.
- **H1 (alternativa):** μ_Δ ≠ 0 (teste bilateral).

Escolha do teste: Shapiro-Wilk nas diferenças pareadas. Se p > 0,05, usa-se o **t de Student pareado**; caso contrário, o **Wilcoxon de postos sinalizados**. O tamanho do efeito é o *d* de Cohen pareado (d_z = média(Δ) / dp(Δ)).

## Resultado principal

| Métrica | H0 (prompt único) | H1 (ensemble) | Δ médio [IC 95%] | Teste | p-valor | d_z |
|---|---|---|---|---|---|---|
| Acurácia Top-1 | 28,21% ± 2,14 | 28,63% ± 2,04 | +0,41 p.p. [0,08; 0,74] | t pareado | **0,0162** | 0,47 |
| F1-Score Macro | 27,59% ± 2,02 | 28,07% ± 2,02 | +0,48 p.p. [0,14; 0,82] | t pareado | **0,0066** | 0,53 |
| Inferência (ms/img) | 24,02 | 24,03 | +0,01 | t pareado | 0,964 | 0,01 |
| Construção do classificador (s) | 0,29 | 2,14 | +1,85 | Wilcoxon | 1,9 × 10⁻⁹ | 7,89 |

**Conclusão: rejeita-se H0 (p = 0,0162 < 0,05).** O Prompt Ensembling melhora a acurácia de forma estatisticamente significativa (H1 venceu em 18 das 30 rodadas, com 2 empates). O ganho absoluto, porém, é pequeno: cerca de 2 acertos a mais a cada 500 imagens. O custo por imagem não muda; o ensemble só custa mais na construção do classificador, que é feita uma única vez.

Acurácia média por tipo de degradação:

| Mudança de domínio | H0 | H1 | Δ (p.p.) |
|---|---|---|---|
| Ruído gaussiano (σ = 0,08) | 22,16% | 22,34% | +0,18 |
| Desfoque gaussiano (σ = 0,7) | 46,73% | 47,08% | +0,36 |
| Cinza + baixo contraste (fator 0,4) | 31,63% | 32,51% | +0,88 |
| Esboço sintético (Sobel invertido) | 12,05% | 12,30% | +0,25 |

## Estrutura do repositório

```
.
├── experimento_clip.ipynb        # Notebook principal, já executado com todas as saídas
├── zeroshot/                     # Código-fonte do experimento
│   ├── config.py                 # Parâmetros: modelo, nº de simulações, shifts, templates, α
│   ├── data.py                   # CIFAR-100, atribuição dos shifts e amostragem estratificada
│   ├── model.py                  # Wrapper do CLIP: embeddings, classificador zero-shot, tempos
│   ├── experiment.py             # Execução das rodadas pareadas e cálculo das métricas
│   ├── stats.py                  # Estatística descritiva, Shapiro-Wilk, t/Wilcoxon, d de Cohen
│   └── plots.py                  # Geração das figuras em 300 DPI
├── resultados/
│   ├── simulacoes_30_rodadas.csv # Métricas brutas: 60 linhas (30 rodadas × 2 cenários)
│   └── resumo_estatistico.json   # Configuração, estatísticas e resultado de cada teste
├── figuras/
│   ├── fig0_exemplos_shift.png   # Imagens originais e as quatro degradações
│   ├── fig1_comparacao_acuracia.png  # Boxplot das 30 acurácias por cenário
│   ├── fig2_evolucao_pareada.png     # Acurácia por rodada e diferença pareada Δ
│   └── fig3_tradeoff_metricas.png    # Médias ± DP de acurácia e F1
├── relatorio_ieee.tex            # Artigo em formato IEEE Conference (coluna dupla)
├── relatorio_ieee.pdf            # Artigo compilado
├── apresentacao_slides.md        # Roteiro da apresentação (7 slides)
└── requirements.txt              # Dependências com versões fixadas
```

## Como executar

### 1. Pré-requisitos

- Python 3.10 ou superior (testado com Python 3.13).
- Cerca de 1 GB livre em disco: o CIFAR-100 (~170 MB) é baixado em `data/` e os pesos do CLIP (~600 MB) vão para o cache do Hugging Face.
- Conexão com a internet na primeira execução.

### 2. Ambiente e dependências

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU (opcional):** o código usa CUDA automaticamente quando disponível. No Windows, o `pip install` padrão instala o PyTorch só para CPU; para usar GPU, instale antes o `torch`/`torchvision` com suporte a CUDA seguindo as instruções de [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Rodar o experimento

**Opção A: interativo.** Abra `experimento_clip.ipynb` no Jupyter ou no VS Code, selecione o kernel do `.venv` e execute todas as células em ordem.

**Opção B: linha de comando.** Executa o notebook inteiro e grava as saídas no próprio arquivo:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 experimento_clip.ipynb
```

O notebook deve ser executado a partir da raiz do repositório, para que o pacote `zeroshot` seja encontrado.

**Tempo estimado:** cerca de 15 minutos em CPU (Intel, 14 threads); bem menos com GPU.

### 4. O que é gerado

| Etapa do notebook | Saída |
|---|---|
| 1. Dataset e mudança de domínio | `figuras/fig0_exemplos_shift.png` |
| 2. Modelo | Informações do ViT-B/32 (49 patches + [CLS], 12 camadas, 151,3 M parâmetros) |
| 3. Estratégias textuais | Templates de H0/H1 e similaridade entre os protótipos |
| 4. 30 simulações | `resultados/simulacoes_30_rodadas.csv` |
| 5. Análise estatística | `resultados/resumo_estatistico.json` e tabelas no notebook |
| 6. Figuras | `figuras/fig1_*.png`, `fig2_*.png`, `fig3_*.png` |
| 7. Conclusão | Decisão sobre H0 impressa no notebook |

Acurácia e F1 são **determinísticos**: uma nova execução reproduz exatamente os mesmos valores. Apenas os tempos variam entre máquinas e execuções.

### 5. Alterar parâmetros

Os parâmetros ficam em `zeroshot/config.py`:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `N_SIMULATIONS` | 30 | Número de rodadas pareadas |
| `SAMPLES_PER_CLASS` | 5 | Imagens por classe em cada rodada (5 × 100 = 500) |
| `ALPHA` | 0,05 | Nível de significância |
| `NOISE_SIGMA`, `BLUR_SIGMA`, `CONTRAST_FACTOR` | 0,08 / 0,7 / 0,4 | Intensidade das degradações |
| `TEMPLATES` | 1 e 8 templates | Prompts de cada cenário |
| `MODEL_ID` | `openai/clip-vit-base-patch32` | Modelo do Hugging Face |
| `BATCH_SIZE` | 128 | Tamanho do lote (reduza se faltar memória) |

Ao mudar algum parâmetro, os CSV, JSON e figuras são sobrescritos, mas os números citados no `.tex` e no roteiro de slides **não** se atualizam sozinhos.

## Relatório IEEE

O PDF compilado (`relatorio_ieee.pdf`, 5 páginas) já está no repositório. Para recompilar:

- **Overleaf:** envie `relatorio_ieee.tex` e a pasta `figuras/` e compile com pdfLaTeX ou XeLaTeX (o preâmbulo detecta o motor).
- **Local:** rode `pdflatex relatorio_ieee.tex` duas vezes, para resolver as referências cruzadas.

## Apresentação

`apresentacao_slides.md` contém o roteiro de 7 slides (cerca de 10 a 12 minutos): problema, as três técnicas, dados e hipóteses, protocolo, resultado principal, consistência e custo, e conclusões e limitações. Cada slide indica o conteúdo, a figura de `figuras/` a usar e a fala principal.

## Referências

- A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.
- A. Radford *et al.*, "Learning transferable visual models from natural language supervision," ICML, 2021.
- R. Bommasani *et al.*, "On the opportunities and risks of foundation models," arXiv:2108.07258, 2021.
