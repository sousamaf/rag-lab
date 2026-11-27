# Experimentos do RAG Lab

| Arquivo | Descrição |
| --- | --- |
| [exp01.py](exp01.py) | Compara o classificador por termos com o classificador por LLM nos mesmos chunks e com o mesmo catálogo. Mede latência e concordância entre métodos. Com gabarito, mede também a qualidade da classificação. |
| [g5_bm25.py](g5_bm25.py) | G5 (BM25): compara variantes e uma grade de `k1`, `b` e IDF com gabarito por trecho; recall por trecho e documento, MRR, latência, repetição e ablação. |
| [g5_topk.py](g5_topk.py) | G5 (BM25): baseline × modo ativo variando `top_k` (5/7/10/12) e sensibilidade dos parâmetros do YAML. |
| [g5_geracao.py](g5_geracao.py) | G5 (BM25): geração com o LLM do YAML para baseline e modo ativo; citações, precisão das fontes, recusas, latência e chamadas, e uma planilha para conferir as respostas. |
| [g5-bm25-notas.md](g5-bm25-notas.md) | G5: corpus versionado, protocolo, resultados, casos em que piorou e decisão. Gabarito: [g5_gold.json](g5_gold.json); configs: [g5-configs/](g5-configs/). |

## Experimento 01

Execute na raiz do projeto. O script reaproveita a configuração, a segmentação e os
classificadores do RAG Lab. Ele executa os dois métodos independentemente de
`RAG_TAG_CLASSIFIER`. A execução usa o endpoint `llm` configurado e pode ser feita
com a API e o PostgreSQL desligados. O corpus padrão é `examples/documents`.

```sh
OLLAMA_BASE_URL=http://127.0.0.1:11434 uv run python experimentos/exp01.py
```

### Avaliação com gabarito

1. Exporte a segmentação do corpus. Essa etapa apenas lê e divide os documentos.

   ```sh
   uv run python experimentos/exp01.py --corpus examples/documents \
     --prepare-gold --output /tmp/exp01-gabarito.json
   ```

2. Anote os trechos antes de consultar as previsões. Use as descrições do catálogo
   para preencher cada `tags` com IDs temáticos. Use `[]` para trechos genuinamente
   sem tema. Mantenha os IDs dos chunks. Para avaliar um corpus do curso, use
   material representativo e revise as divergências entre anotadores.
3. Execute os dois métodos com o mesmo corpus e a mesma configuração de segmentação.

   ```sh
   OLLAMA_BASE_URL=http://127.0.0.1:11434 uv run python experimentos/exp01.py \
     --corpus examples/documents --gold /tmp/exp01-gabarito.json \
     --output /tmp/exp01-resultado.json
   ```

O JSON inclui tags e tempo por chunk, concordância exata entre métodos, mediana e
tempo total por método. Com gabarito, inclui também precisão, revocação e F1 micro
por par trecho–rótulo, além da proporção de chunks cujo conjunto de tags coincide
com o gabarito. A concordância entre os métodos mede igualdade das previsões.

O tempo cobre as chamadas de classificação e inclui rede e eventual carregamento
inicial do LLM. Registre modelo, versão, catálogo, hardware e condições de execução.
Para comparar desempenho, repita o ensaio em condições equivalentes e use o mesmo
gabarito. O arquivo de saída inclui o texto dos trechos, as previsões e o hash do
catálogo usado.
