# G5 — BM25: comparação de parâmetros e IDF

Reprodução: [g5_bm25.py](g5_bm25.py) com o gabarito [g5_gold.json](g5_gold.json).
Saída em `data/results/g5/` (ignorada pelo Git): `manifesto.json` (commit, data, hardware,
versões, hash do gabarito, estatísticas do corpus), `<variante>/config.yaml` + um envelope por
pergunta, `resumo.json`, `perguntas.csv` (posição do 1º relevante, recall e latência por pergunta e
variante), `grade.csv` e `ablacao.json`. O manifesto grava o commit e se havia alterações não
commitadas: para a entrega, rode o script de novo depois de commitar.

## Corpus e perguntas

O corpus é local (`data/documents/*` é ignorado pelo Git, como o professor definiu). Para
recriá-lo: 12 `.txt` em português, **uma sentença por linha lógica**, ~5 sentenças cada
(61 trechos, 35 a 210 caracteres, mediana ~104): `bm25`, `embeddings`, `ingestao`, `avaliacao`
(núcleo da primeira rodada) e `tokenizacao`, `indice_invertido`, `tfidf`, `chunking`,
`alucinacao`, `banco_dados`, `historia_busca` e `culinaria` (fora do tema, serve de distrator).
Há termos repetidos entre documentos (ex.: "índice", "tempo", "token") para gerar confusão
lexical, e trechos curtos e longos para dar o que normalizar ao `b`. Ingestão com
`RAG_TAG_CLASSIFIER=terms` e Embedding desligado.

Gabarito escrito a partir dos textos antes de rodar (não usa a saída do sistema):
20 perguntas — 5 literais,
5 semânticas (parafraseadas), 5 compostas e 5 sem resposta (3 *near-miss* que citam o assunto
sem a informação, 2 fora do domínio). Relevantes são marcados por trecho (`arquivo:posição`) e cada pergunta traz `deve_conter`, o que uma
resposta sustentada precisa dizer (nas sem resposta: reconhecer a falta de evidências).

## Protocolo

Só o BM25 ativo; geração desativada (mede recuperação, não o LLM); `top_k=5`, `candidate_k=12`.
Uma variável por vez a partir da baseline (IDF `lucene`, `k1=1.5`, `b=0.75`) e uma grade
completa `k1 ∈ {0.5, 0.9, 1.2, 1.5, 2, 3}` × `b ∈ {0, .25, .5, .75, 1}` × IDF `{lucene, okapi}`.
Métricas: recall@5 por trecho e por documento, MRR, latência do módulo (do `trace`); nas
perguntas sem resposta, fração com algum resultado e score do 1º.

## Resultados (variantes nomeadas)

| Variante | perfil | recall trecho | recall doc | MRR |
|---|---|---|---|---|
| baseline | literal | 0.80 | 1.00 | 0.80 |
| | semântica | 0.73 | 0.83 | 0.60 |
| | composta | 0.67 | 0.80 | 0.80 |
| b=0 | literal | 0.80 | **0.80** | 0.80 |
| | semântica | 0.73 | 0.83 | **0.70** |
| | composta | 0.67 | 0.80 | **0.70** |
| idf okapi | composta | **0.73** | **0.90** | 0.80 |
| b=1, k1=0.5, k1=3 | — | ≈ baseline | ≈ baseline | ≈ baseline |

Sem resposta: **100% das 5 perguntas devolvem trechos em todas as variantes** (score do 1º entre
5.7 e 7.2 em média; ver abaixo).

**Grade completa (60 combinações):** MRR entre 0.70 e 0.77 (amplitude 0.067); recall por trecho
entre 0.73 e 0.79. Melhor: IDF okapi, `k1=0.5`, `b=0`; pior: lucene, `k1=0.9`, `b=0.25`.
Repetir a baseline devolve trechos e scores idênticos. Ablação: BM25 desligado → `bypass` no
trace, sem candidatos.

## Posição do 1º relevante (top-5, `-` = não apareceu)

Idêntica em todas as variantes, exceto: **S5** (baseline 2, okapi 3, `b=0` 1, `b=1` 3, `k1=0.5` 2,
`k1=3` 2) e **C5** (`b=0` cai de 1 para 2). L5 e S4 ficam sem relevante no top-5 em qualquer
variante. Uma única pergunta (S5) é a que mais se move com os parâmetros.

## Casos concretos (baseline)

- **L5 "O que é uma stopword?"** → perdeu `tokenizacao:1`: o texto diz "Stopwords" (plural);
  sem stemming o termo da consulta não casa. Nenhuma variante de `k1`/`b`/IDF resolve.
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
  É o pior tipo de erro: confiante e errado. **N4 "assar pão"** → `culinaria:3` (7.3), o texto é
  de bolo.
- **Piora com variante (guardar):** `b=0` melhora o MRR semântico (0.60 → 0.70), mas piora a
  composta (0.80 → 0.70) e o recall por documento nas literais (1.00 → 0.80). Visto em S1: com
  `b=0` o tamanho médio do top-5 sobe de 21.4 para 29.8 termos (entram `banco_dados:2` e
  `chunking:2`, longos; saem `chunking:4` e `embeddings:0`, curtos). Em duas outras perguntas
  conferidas o tamanho médio não mudou, então o efeito não é geral. Ganho em um perfil,
  perda em outro, sem melhora consistente.

## Decisão: manter ou rejeitar

- **Manter** o BM25 como recuperador lexical com os padrões `idf: lucene`, `k1=1.5`, `b=0.75`:
  nenhuma variante testada melhora o conjunto de forma consistente (MRR 0.70–0.77 em 60
  combinações; `b=0` ganha em S1/S5 e perde em literais e C5).
- **Manter `idf: lucene`** como padrão: é o único que nunca dá score ≤ 0 em corpus mínimo
  (teste de um documento); `okapi` fica como opção para comparação.
- **Rejeitar, por ora, qualquer recomendação de valor diferente** para `k1`/`b`: o efeito está
  dentro do ruído de 20 perguntas e 61 trechos.
- Os erros que sobram (L5, S4, perguntas sem resposta) não são de parâmetro e precisam de outra
  técnica (Embedding/Hybrid) ou de tratamento de consulta.

## Fora desta rodada

Não foram medidos fidelidade da resposta às fontes, citações corretas, chamadas aos modelos
(aqui são 0: só BM25, sem LLM) e variabilidade do LLM, porque a geração está desativada. Ficam
para quando houver endpoint de modelo configurado no YAML.

## Conclusões (e limites)

1. **`k1`, `b` e IDF mexem pouco** (MRR varia 0.067 em 60 combinações; 20 perguntas, 61 trechos).
   Diferenças de uma ou duas perguntas não são evidência para recomendar um valor; o padrão
   `k1=1.5`, `b=0.75`, `lucene` é defensável.
2. **O que limita o BM25 aqui não são os parâmetros**: forma da palavra (stopword/stopwords),
   vocabulário diferente (S4, S1), letras soltas ("b") e perguntas sem resposta. Isso aponta para
   tokenização/stemming e limiar de score — fora do alcance dos parâmetros.
3. **Sem resposta:** os scores de resultados espúrios (2.5–8.6) se sobrepõem aos de acertos
   (3.8–15), então um limiar global único cortaria acertos; precisa ser avaliado.
4. Limites: corpus pequeno e escrito pelo grupo; latências de ~10–18 ms variam entre execuções e
   não distinguem variantes; sem LLM, nada se afirma sobre fidelidade da resposta final.
