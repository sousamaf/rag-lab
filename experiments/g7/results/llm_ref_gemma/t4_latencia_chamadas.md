# Tabela T4 — Latência por Etapa e Chamadas

| config | n_consultas | hybrid_status | hybrid_ms_mediana | hybrid_ms_p95 | tfidf_ms_mediana | bm25_ms_mediana | embedding_ms_mediana | reranking_ms_mediana | generation_ms_mediana | generation_ms_p95 | wall_time_ms_mediana | wall_time_ms_p95 | chat_calls_total | embed_calls_total | embed_texts_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_rr_bm25_emb | 24 | bypass | 0.002 | 0.003 | 0.001 | 41.323 | 68.254 | 0.001 | 10535.449 | 24418.601 | 10652.422 | 24508.487 | 24 | 24 | 24 |
| e2_rrf_k60_bm25_emb | 24 | executed | 0.99 | 2.048 | 0.0 | 32.99 | 67.65 | 0.001 | 9942.653 | 25405.419 | 10050.62 | 25484.802 | 24 | 24 | 24 |
