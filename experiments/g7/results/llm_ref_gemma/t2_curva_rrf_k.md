# Tabela T2 — Replay Offline: Curva rrf_k, Permutações Round-Robin e Estratégias

| familia_replay | variante | rrf_k | regime_dominancia | recuperadores | n_perguntas | doc_recall_5 | doc_mrr | doc_ndcg_5 | trecho_recall_1 | trecho_recall_3 | trecho_recall_5 | trecho_recall_10 | trecho_mrr | trecho_ndcg_5 | boundary_tie_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2_curva_rrf_k | rrf_k=1 | 1 | k<=10 (1o isolado pode vencer m=2) | bm25+embedding | 18 | 0.9167 | 0.8704 | 0.8605 | 0.3796 | 0.8056 | 0.8889 | 1.0 | 0.8009 | 0.7802 | 0.1667 |
| E2_curva_rrf_k | rrf_k=3 | 3 | k<=10 (1o isolado pode vencer m=2) | bm25+embedding | 18 | 0.9722 | 0.8907 | 0.8928 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8246 | 0.7871 | 0.0556 |
| E2_curva_rrf_k | rrf_k=5 | 5 | k<=10 (1o isolado pode vencer m=2) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1111 |
| E2_curva_rrf_k | rrf_k=10 | 10 | k<=10 (1o isolado pode vencer m=2) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=11 | 11 | 10<k<=21 (m=2 domina m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=15 | 15 | 10<k<=21 (m=2 domina m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=21 | 21 | 10<k<=21 (m=2 domina m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=22 | 22 | k>21 (dominancia cardinal m=3>m=2>m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=30 | 30 | k>21 (dominancia cardinal m=3>m=2>m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=60 | 60 | k>21 (dominancia cardinal m=3>m=2>m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=120 | 120 | k>21 (dominancia cardinal m=3>m=2>m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E2_curva_rrf_k | rrf_k=1000 | 1000 | k>21 (dominancia cardinal m=3>m=2>m=1) | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E3_permutacao_round_robin | rr(bm25->embedding) | - | sem_fusao_round_robin | bm25->embedding | 18 | 1.0 | 0.875 | 0.8976 | 0.3796 | 0.7593 | 0.963 | 1.0 | 0.8333 | 0.8149 | 0.0 |
| E3_permutacao_round_robin | rr(embedding->bm25) | - | sem_fusao_round_robin | embedding->bm25 | 18 | 1.0 | 0.9352 | 0.936 | 0.4074 | 0.8981 | 0.963 | 1.0 | 0.8426 | 0.8363 | 0.0 |
| E4_estrategias_fusao | rrf | 60 | estrategia_hibrida | bm25+embedding | 18 | 0.9722 | 0.8843 | 0.8908 | 0.4074 | 0.7778 | 0.8889 | 1.0 | 0.8144 | 0.7815 | 0.1667 |
| E4_estrategias_fusao | weighted_rrf | 60 | estrategia_hibrida | bm25+embedding | 18 | 1.0 | 0.8843 | 0.904 | 0.4074 | 0.75 | 0.9259 | 1.0 | 0.8194 | 0.804 | 0.0 |
| E4_estrategias_fusao | convex | - | estrategia_hibrida | bm25+embedding | 18 | 0.9167 | 0.8981 | 0.881 | 0.463 | 0.8426 | 0.9074 | 0.9815 | 0.8889 | 0.8301 | 0.0 |
| E4_estrategias_fusao | dbsf | - | estrategia_hibrida | bm25+embedding | 18 | 0.9722 | 0.8889 | 0.8931 | 0.4352 | 0.7778 | 0.9074 | 0.9815 | 0.8514 | 0.8106 | 0.0 |
