# Roteiro de Apresentação: Prompt Ensembling no CLIP sob Mudança de Domínio

Duração sugerida: 10 a 12 minutos (cerca de 1min30 por slide).
Todos os números vêm de `resultados/resumo_estatistico.json` e `resultados/simulacoes_30_rodadas.csv`.

---

## Slide 1: Problema e motivação

**Título:** Classificação zero-shot sob mudança de domínio: o texto do prompt faz diferença?

- Paradigma especialista (CNNs): um modelo por tarefa, dados rotulados e retreinamento a cada novo conjunto de classes.
- Modelos de fundação (Bommasani et al., 2021): pré-treino amplo e adaptação a muitas tarefas.
- O CLIP classifica imagens comparando-as com frases ("a photo of a dog"), sem treinamento adicional.
- Pergunta: **a forma de escrever o prompt melhora a robustez quando as imagens fogem da distribuição de treino?**

**Visual:** diagrama simples: imagem → encoder visual; frases das classes → encoder de texto; similaridade de cosseno → classe.

**Fala-chave:** "No CLIP, mudar o texto muda o classificador. Queremos saber se isso ajuda quando a imagem está degradada."

---

## Slide 2: As três técnicas

| Técnica | O que é | Papel no experimento |
|---|---|---|
| A. ViT-B/32 | Imagem 224×224 dividida em 49 patches de 32×32, mais o token [CLS]; 12 camadas Transformer | Backbone visual (fixo) |
| B. CLIP | Encoders de imagem e texto alinhados por aprendizado contrastivo em 400M pares; espaço conjunto de 512 dimensões | Classificador zero-shot por similaridade de cosseno |
| C. Inferência textual | Prompt único vs. média L2-normalizada de 8 templates | **Variável experimental** |

- Fórmula do ensemble: w_c = normalizar( média_k ( normalizar( g(template_k(c)) ) ) )
- Predição: ŷ = argmax_c cos(v, w_c)

**Visual:** tabela acima e a fórmula do protótipo.

---

## Slide 3: Dados, mudança de domínio e hipóteses

- CIFAR-100, conjunto de teste: 10.000 imagens, 100 classes.
- Quatro mudanças de domínio, 2.500 imagens cada: ruído gaussiano (σ = 0,08), desfoque (σ = 0,7), cinza com contraste de 40% e esboço sintético (Sobel invertido).
- **H0:** prompt único "a photo of a {classe}" (μ_Δ = 0, sem efeito).
- **H1:** Prompt Ensembling com 8 templates (μ_Δ ≠ 0).

**Gráfico:** `figuras/fig0_exemplos_shift.png` (original e as quatro degradações).

**Fala-chave:** "O esboço é a degradação mais agressiva: a acurácia cai para cerca de 12%."

---

## Slide 4: Protocolo experimental

- 30 simulações, sementes 1 a 30.
- Em cada rodada: 500 imagens estratificadas (5 por classe).
- **Pareamento estrito:** H0 e H1 avaliados sobre as mesmas 500 imagens.
- Ordem de execução alternada por semente e rodada de aquecimento descartada (justiça nas medidas de tempo).
- Métricas: Acurácia Top-1, F1-Score Macro, tempo de inferência por imagem e tempo de construção do classificador.
- Estatística: Shapiro-Wilk nas diferenças → t pareado (se normal) ou Wilcoxon (se não); α = 0,05; d de Cohen pareado.

**Visual:** fluxograma: semente i → amostra estratificada → [H0 | H1] → métricas → teste pareado.

---

## Slide 5: Resultado principal

| Métrica | H0 | H1 | Δ médio [IC 95%] | Teste | p-valor | d_z |
|---|---|---|---|---|---|---|
| Acurácia Top-1 | 28,21% ± 2,14 | 28,63% ± 2,04 | +0,41 p.p. [0,08; 0,74] | t pareado | **0,0162** | 0,47 |
| F1-Score Macro | 27,59% ± 2,02 | 28,07% ± 2,02 | +0,48 p.p. [0,14; 0,82] | t pareado | **0,0066** | 0,53 |

- Shapiro-Wilk nas diferenças de acurácia: W = 0,9505, p = 0,174 → normal → t pareado.
- t(29) = 2,554, p = 0,0162 < 0,05 → **rejeitamos H0**.

**Gráfico:** `figuras/fig1_comparacao_acuracia.png` (boxplot com as 30 rodadas e o p-valor no título).

**Fala-chave:** "A diferença é estatisticamente significativa, mas é pequena: cerca de 2 acertos a mais em cada 500 imagens."

---

## Slide 6: Consistência, efeito por domínio e custo

- H1 vence em **18 de 30** rodadas, empata em 2 e perde em 10 (Δ entre −1,2 e +1,8 p.p.).
- Ganho por tipo de degradação (acurácia H0 → H1):
  - Cinza e baixo contraste: 31,63% → 32,51% (**+0,88 p.p.**, o maior ganho)
  - Desfoque: 46,73% → 47,08% (+0,36)
  - Esboço: 12,05% → 12,30% (+0,25)
  - Ruído: 22,16% → 22,34% (+0,18)
- Custo: inferência por imagem igual (24,02 vs. 24,03 ms; t pareado p = 0,964). Construção do classificador: 0,29 s → 2,14 s (Wilcoxon p = 1,9 × 10⁻⁹), custo pago uma única vez.

**Gráficos:** `figuras/fig2_evolucao_pareada.png` (principal) e `figuras/fig3_tradeoff_metricas.png` (apoio, canto do slide).

**Fala-chave:** "O ensemble ajuda mais onde um template descreve literalmente a degradação, o que também é uma limitação do desenho."

---

## Slide 7: Conclusões e limitações

**Conclusões**
- Prompt Ensembling melhora a acurácia zero-shot do CLIP sob mudança de domínio de forma estatisticamente significativa (p = 0,0162), com ganho absoluto pequeno (+0,41 p.p.).
- É uma melhoria de baixo risco: sem treinamento e sem custo extra por imagem.
- Não resolve a mudança de domínio: a acurácia continua abaixo de 30%.

**Limitações**
- Rodadas sorteadas do mesmo conjunto de 10.000 imagens (não totalmente independentes).
- Degradações sintéticas em imagens de 32×32 ampliadas para 224×224.
- Alguns templates descrevem as próprias degradações.
- CLIP: composicionalidade fraca e vieses dos dados da web (ex.: nomes ambíguos como "mouse", "ray").

**Trabalhos futuros:** templates neutros como controle, mudanças de domínio naturais, ViT-B/16 e ViT-L/14, prompts aprendidos com poucos exemplos.

**Visual:** apenas texto, com destaque para o p-valor e o Δ.
