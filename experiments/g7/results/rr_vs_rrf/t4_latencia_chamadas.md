# Tabela T4 — Latência por Etapa e Chamadas

| config | n_consultas | hybrid_status | hybrid_ms_mediana | hybrid_ms_p95 | tfidf_ms_mediana | bm25_ms_mediana | embedding_ms_mediana | reranking_ms_mediana | generation_ms_mediana | generation_ms_p95 | wall_time_ms_mediana | wall_time_ms_p95 | chat_calls_total | embed_calls_total | embed_texts_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_rr_bm25_emb | 24 | bypass | 0.002 | 0.002 | 0.001 | 29.069 | 53.493 | 0.001 | 5272.57 | 8438.071 | 5368.221 | 8551.514 | 24 | 24 | 24 |
| e1_rr_todos | 24 | bypass | 0.002 | 0.002 | 25.526 | 31.773 | 63.076 | 0.001 | 4962.577 | 9939.935 | 5075.385 | 10067.399 | 24 | 24 | 24 |
| e2_rrf_k10_bm25_emb | 24 | executed | 0.99 | 1.635 | 0.001 | 30.387 | 59.15 | 0.001 | 3754.122 | 8000.582 | 3850.173 | 8091.632 | 24 | 24 | 24 |
| e2_rrf_k120_bm25_emb | 24 | executed | 1.035 | 1.399 | 0.0 | 31.459 | 54.907 | 0.001 | 3819.88 | 10418.123 | 3926.161 | 10503.687 | 24 | 24 | 24 |
| e2_rrf_k1_bm25_emb | 24 | executed | 0.909 | 1.609 | 0.0 | 31.476 | 56.784 | 0.001 | 5360.678 | 11935.581 | 5461.016 | 12014.638 | 24 | 24 | 24 |
| e2_rrf_k60_bm25_emb | 24 | executed | 1.603 | 1.976 | 0.001 | 41.241 | 63.805 | 0.002 | 3559.392 | 10010.043 | 3658.837 | 10101.887 | 24 | 24 | 24 |
