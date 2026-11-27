# G5 — BM25: comparação de parâmetros e IDF

Reprodução: [g5_bm25.py](g5_bm25.py) (variantes e grade), [g5_topk.py](g5_topk.py) (baseline ×
modo ativo por `top_k` e sensibilidade) e [g5_geracao.py](g5_geracao.py) (geração e citações),
todos com o gabarito [g5_gold.json](g5_gold.json) e as configs de [g5-configs/](g5-configs/).
Saída em `data/results/g5/` (ignorada pelo Git): `manifesto.json` (commit, data, hardware,
versões, hash do gabarito, estatísticas do corpus), `<variante>/config.yaml` + um envelope por
pergunta, `resumo.json`, `perguntas.csv` (posição do 1º relevante, recall e latência por pergunta e
variante), `grade.csv`, `ablacao.json` e `topk/`. O manifesto grava o commit e se havia
alterações não commitadas: para a entrega, rode os scripts de novo depois de commitar.

## Corpus e perguntas

O corpus está versionado em [g5-corpus/](g5-corpus/): 12 `.txt` em português, uma sentença
por linha lógica, ~5 sentenças cada (61 trechos, 35 a 210 caracteres, mediana ~104): `bm25`,
`embeddings`, `ingestao`, `avaliacao`, `tokenizacao`, `indice_invertido`, `tfidf`, `chunking`,
`alucinacao`, `banco_dados`, `historia_busca` e `culinaria` (fora do tema, serve de distrator).
Há termos repetidos entre documentos (ex.: "índice", "tempo", "token") para gerar confusão
lexical, e trechos curtos e longos para dar o que normalizar ao `b`. Ingestão:
`RAG_TAG_CLASSIFIER=terms rag-lab --config experimentos/g5-configs/baseline.yaml ingest
experimentos/g5-corpus` (Embedding desligado).

Gabarito escrito a partir dos textos antes de rodar (não usa a saída do sistema): 30 perguntas —
8 literais, 8 semânticas (parafraseadas), 8 compostas e 6 sem resposta (5 *near-miss* que citam o
assunto sem a informação e 1 fora do domínio). Relevantes são marcados por trecho
(`arquivo:posição`) e cada pergunta traz `deve_conter`, o que uma resposta sustentada precisa
dizer (nas sem resposta: reconhecer a falta de evidências).

## Protocolo

Só o BM25 ativo; as tabelas de recuperação usam geração desativada; `top_k=5`, `candidate_k=12`.
Uma variável por vez a partir da baseline (IDF `lucene`, `k1=1.5`, `b=0.75`, `min_idf=0`) e uma
grade completa `k1 ∈ {0.5, 0.9, 1.2, 1.5, 2, 3}` × `b ∈ {0, .25, .5, .75, 1}` × IDF
`{lucene, okapi, classic}`; `min_idf` (ignora termos da pergunta com IDF baixo) entra como
variante nomeada e na escolha do modo ativo. Métricas: recall@5 por trecho e por documento, MRR,
latência do módulo (do `trace`); nas perguntas sem resposta, fração com algum resultado e score do 1º.

## Resultados (variantes nomeadas, 30 perguntas)

| Variante | perfil | recall trecho | recall doc | MRR |
|---|---|---|---|---|
| baseline | literal | 0.88 | 1.00 | 0.88 |
| | semântica | 0.83 | 0.90 | 0.69 |
| | composta | 0.79 | 0.88 | 0.88 |
| b=0 | literal | 0.88 | **0.88** | **0.81** |
| | semântica | 0.83 | 0.90 | 0.71 |
| | composta | **0.73** | 0.88 | **0.81** |
| idf okapi | composta | 0.77 | **0.94** | 0.88 |
| min_idf=1.0 | literal | 0.88 | 1.00 | **0.81** |
| | semântica | 0.83 | **0.96** | 0.67 |
| | composta | 0.77 | 0.88 | 0.88 |
| min_idf=1.5 | literal | 0.88 | **0.88** | **0.81** |
| | semântica | 0.83 | 0.90 | **0.73** |
| | composta | **0.83** | **0.94** | **0.94** |
| b=1, k1=0.5, k1=3 | — | ≈ baseline | ≈ baseline | ≈ baseline |

Sem resposta: 100% das 6 perguntas devolvem trechos em todas as variantes, **exceto `min_idf=1.5`
(83%, uma pergunta a menos)**. Score do 1º resultado: 6.2 na baseline e 4.95 com `min_idf=1.5`.

**Grade k1 × b × IDF (90 combinações):** MRR entre 0.76 e 0.85 (amplitude 0.087). Melhor:
okapi, `k1=0.9`, `b=1`; pior: lucene, `k1=0.9`, `b=0.25`. Repetir a baseline devolve trechos e
scores idênticos. Ablação: BM25 desligado → `bypass` no trace, sem candidatos.

## Posição do 1º relevante (top-5, `-` = não apareceu)

Igual à baseline na maioria das perguntas. Mudam: **S5** (baseline 2; okapi 3; `b=0` 1; `b=1` 3;
`min_idf` 3), **S7** (baseline 2; `b=0` 5; `b=1` 1; `k1=0.5` 3), **L3** (`min_idf` 1.0/1.5: 1 → 2),
**L6** e **C5** (`b=0`: 1 → 2), **C1** (`min_idf=1.0`: 1 → 2), **C2** (`min_idf`: 2 → 1) e **S1**
(`min_idf=1.5`: 2 → 1). L5 e S4 ficam sem relevante no top-5 em qualquer variante.

## Casos concretos (baseline)

- **L5 "O que é uma stopword?"** → perdeu `tokenizacao:1`: o texto diz "Stopwords" (plural);
  sem stemming o termo da consulta não casa. Nenhuma variante de `k1`/`b`/IDF/`min_idf` resolve.
- **S4 "dividir em pedaços pequenos atrapalha?"** → nenhum relevante no top-5; os trechos
  (`chunking:1`, `ingestao:3`) falam de "trechos curtos", vocabulário diferente da pergunta.
- **C1 "diferença entre k1 e b e por que o IDF importa?"** → só `bm25:3` no topo (7.74);
  `bm25:2` fica em 9º (4.92) e `bm25:4` em 13º (4.03). Verificado termo a termo: "b" **não** é o
  problema (df=2, IDF 3.2, dá 4.9 pontos a `bm25:3`). O que atrapalha: "diferença" e "importa" não
  existem no corpus (df=0) e a pergunta tem muitas palavras funcionais ("a", "e", "o", "que",
  "por") que somam pontos em trechos sem relação, como `culinaria:1`, que é curto (10 termos) e
  chega a 5.62. `bm25:2` só pontua por "k1" (3.68) e `bm25:4` por "idf" (1.96).
- **N2 "valor padrão de k1 no Elasticsearch?" (sem resposta)** → `historia_busca:3` com score
  **8.6**, mais alto que vários acertos legítimos; o trecho cita Elasticsearch mas não o valor.
  **N6 "preço da licença do Elasticsearch"** → o mesmo trecho com 7.1; **N4 "assar pão"** →
  `culinaria:3` (7.3), o texto é de bolo. É o pior tipo de erro: confiante e errado, e `min_idf`
  não o corrige porque as palavras que casam ("Elasticsearch", "forno") são raras.
- **Piora com variante (guardar):** `b=0` piora a composta (MRR 0.88 → 0.81), o recall por
  documento nas literais (1.00 → 0.88) e leva S7 do 2º para o 5º lugar. Visto em S1: com `b=0` o
  tamanho médio do top-5 sobe de 21.4 para 29.8 termos (entram `banco_dados:2` e `chunking:2`,
  longos; saem `chunking:4` e `embeddings:0`, curtos); em duas outras perguntas conferidas o
  tamanho médio não mudou, então o efeito não é geral. `min_idf` ajuda a composta (0.88 → 0.94)
  e piora literais (L3 cai para o 2º lugar): ganho em um perfil, perda em outro.

## Decisão: manter ou rejeitar

- **Manter** o BM25 como recuperador lexical com os padrões `idf: lucene`, `k1=1.5`, `b=0.75` e
  `min_idf=0` (desligado): nenhuma variante testada melhora o conjunto de forma consistente (MRR
  0.76–0.85 em 90 combinações; ganhos em um perfil vêm com perdas em outro).
- **Manter `idf: lucene`** como padrão: é o único que nunca dá score ≤ 0 em corpus mínimo
  (teste de um documento); `okapi`, `classic` e `min_idf` ficam como opções para comparação.
- **Rejeitar, por ora, qualquer recomendação de valor diferente** para `k1`/`b`: o efeito está
  dentro do ruído de 30 perguntas e 61 trechos.
- Os erros que sobram (L5, S4, perguntas sem resposta com termo raro) não são de parâmetro e
  precisam de outra técnica (Embedding/Hybrid) ou de tratamento de consulta.

## Baseline × modo ativo variando o `top_k` (método de avaliação do professor)

Como no G7: rodar a baseline com o valor padrão, variar o parâmetro e repetir com o modo ativo, na
mesma tabela. Script: [g5_topk.py](g5_topk.py), saída em `data/results/g5/topk/`.

- **Baseline** = BM25 de referência (IDF `lucene`, `k1=1.5`, `b=0.75`, `min_idf=0`).
- **Parâmetro variado:** `pipeline.top_k` ∈ {5, 7, 10, 12} (o limite é 12 porque `candidate_k`
  padrão é 12 e a validação exige `candidate_k ≥ top_k`). Métricas sobre as evidências finais.
- **Modo ativo** = combinação de `k1`, `b`, IDF e `min_idf` (450 combinações) escolhida pelo MRR na
  metade "treino" das perguntas (metade de cada perfil; desempate: mais perto da referência).
  Resultado: `idf: classic`, `k1=0.9`, `b=0.25`, `min_idf=2.0` ([ativo.yaml](g5-configs/ativo.yaml)).
  A outra metade ("teste", 12 perguntas com resposta e 3 sem) não participou da escolha.

| top_k | modo | MRR todas | MRR teste | recall trecho todas | recall doc todas | sem resposta c/ evidência |
|---|---|---|---|---|---|---|
| 5 | baseline | 0.812 | 0.833 | 0.833 | 0.924 | 100% |
| 5 | ativo | 0.833 | 0.792 | 0.826 | 0.861 | 83% |
| 7 | baseline | 0.812 | 0.833 | 0.854 | 0.965 | 100% |
| 7 | ativo | 0.833 | 0.792 | 0.847 | 0.917 | 83% |
| 10 | baseline | 0.812 | 0.833 | 0.868 | 1.000 | 100% |
| 10 | ativo | 0.833 | 0.792 | 0.882 | 0.958 | 83% |
| 12 | baseline | 0.812 | 0.833 | 0.868 | 1.000 | 100% |
| 12 | ativo | 0.833 | 0.792 | 0.882 | 0.958 | 83% |

**Leitura:** nas 30 perguntas o modo ativo ganha 0.021 de MRR e corta uma pergunta sem resposta
(`min_idf=2.0` descarta os termos comuns), ao custo de recall por documento (0.924 → 0.861 em
`top_k=5`). Nas perguntas de teste, que não participaram da escolha, o MRR **cai** (0.833 →
0.792) e o recall por trecho fica igual. Conclusão: **não demonstramos que o modo ativo seja
melhor que a baseline**; o ganho nas 30 perguntas vem das usadas na escolha. Ficar no mesmo valor
da baseline é um resultado aceitável para o método de avaliação do professor. A melhor
combinação escolhida olhando todas as perguntas (`classic`, `k1=1.5`, `b=1`, `min_idf=2.0`) chega a
MRR 0.868, mas esse número é otimista porque foi escolhido nos mesmos dados.

## Quais parâmetros do YAML impactam um BM25 sozinho

Nº de perguntas (de 30) cujas evidências mudam em relação à baseline com `top_k=5`:

| Parâmetro | Efeito |
|---|---|
| `pipeline.top_k` | 30/30 mudam ao passar de 5 para 7, 10 ou 12 (define quantas fontes seguem) |
| `pipeline.max_context_chars` | 12000 → 0; 600 → 20; 300 e 150 → 30 (orçamento corta evidências) |
| `modules.bm25.b` | 0 → 28; 0.25 → 26; 0.5 → 19; 1 → 21 |
| `modules.bm25.k1` | 0.5 → 19; 0.9 → 12; 1.2 → 8; 2 → 12; 3 → 21 |
| `modules.bm25.idf` | okapi → 24; classic → 11 |
| `modules.bm25.min_idf` | 1.0 → 23; 1.2 → 28; 1.5 → 28; 2.0 → 29 |
| `pipeline.candidate_k` | 5, 12 e 30 → **0** (com uma só consulta e `top_k=5` não muda nada) |

`candidate_k` só importa quando há várias consultas (Multi-query/Decomposição) ou fusão, que não
estão ativas aqui. `sentence_radius`, `expanded_queries` e `rrf_k` pertencem a outros módulos.
Mudar a ordem das evidências conta como mudança, mesmo sem alterar o conjunto.

## Geração, citações, latência e chamadas

[g5_geracao.py](g5_geracao.py) executa o pipeline completo com o LLM do YAML para as configs
`experimentos/g5-configs/baseline.yaml` e `ativo.yaml`, no mesmo molde das tabelas T3 e T4 do G7:

- **T3 — geração e citações:** taxa de respostas com citação `[n]`, sintaxe válida (n entre 1 e o
  número de evidências), precisão das fontes citadas por documento e por trecho, cobertura
  lexical dos fatos de `deve_conter`, abstenção nas perguntas sem resposta e o status da rubrica
  humana de fidelidade (0–2).
- **T4 — latência e chamadas:** mediana e p95 do tempo do BM25, da geração e total, e o número de
  chamadas ao chat e aos embeddings.
- `anotacao_humana.csv`: modelo para a rubrica de fidelidade e a conferência manual das citações.

## Conclusões (e limites)

1. **`k1`, `b` e IDF mexem pouco** (MRR varia 0.087 em 90 combinações; 30 perguntas, 61 trechos).
   Diferenças de uma ou duas perguntas não são evidência para recomendar um valor; o padrão
   `k1=1.5`, `b=0.75`, `lucene` é defensável.
2. **`min_idf` é a única opção que reduz resultados espúrios**, mas só em perguntas cujas palavras
   que casam são comuns; não resolve termos raros fora de contexto (N2, N4, N6) e custa recall.
3. **O que limita o BM25 aqui não são os parâmetros**: forma da palavra (stopword/stopwords),
   vocabulário diferente (S4, S1), letras soltas ("b") e perguntas sem resposta. Isso aponta para
   tokenização/stemming — fora do alcance dos parâmetros.
4. Limites: corpus pequeno e escrito pelo grupo; latências de ~8–20 ms variam entre execuções e
   não distinguem variantes.
