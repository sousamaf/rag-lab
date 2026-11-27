# G7 · Hybrid RAG — Arquitetura, Controle Experimental, Comparação Round-Robin vs. RRF e Limitações

> **Grupo G7 — Hybrid (`src/rag_lab/modules/hybrid.py`)**  
> **Integrantes:** Diego · Gabriel Alves · João Victor Póvoa  
> **Repositório:** `rag-lab` · **Módulo principal:** [`src/rag_lab/modules/hybrid.py`](../src/rag_lab/modules/hybrid.py)  
> **Referência de contrato e plano:** [`src/rag_lab/config.py`](../src/rag_lab/config.py) · [`docs/g7-plano-implementacao.md`](g7-plano-implementacao.md)

---

## Sumário Executivo

Este relatório consolida a fundamentação teórica, o desenho de engenharia, a auditoria de desligamento seguro (`hybrid: false`), os testes de regressão e a avaliação experimental controlada do módulo **Hybrid** (**G7**) no `rag-lab`. Em atendimento integral às diretrizes da revisão do PR #6 pelo Prof. Marco Antonio Firmino de Sousa, todo o protocolo empírico foi conduzido **sem confundir modelos**:

1. **Banco e índice vetorial único e fixo:** ingestão única dos `12` documentos autorais (`135` chunks sentenciais em `20,72 s`) no banco `rag_g7` com **`embeddinggemma:300m`** (`768` dimensões, `RAG_TAG_CLASSIFIER=terms`).
2. **Bateria principal controlada (`rr_vs_rrf`, 144 consultas):** comparação direta entre *round-robin* (`hybrid: false`, `status: bypass`) e **RRF** variando `rrf_k` $\in \{1, 10, 60, 120\}$ com o par fixo **`llama3.2:3b` + `embeddinggemma:300m`**, além de **replay offline exato** (`144/144` envelopes com `all_matched: true`) cobrindo `rrf_k` $\in [1, 1000]$, todas as permutações de *round-robin* e as estratégias `weighted_rrf`, `convex` e `dbsf`.
3. **Referência pareada de LLM sobre o mesmo embedding (`llm_ref_gemma`, 48 consultas):** comparação isolada entre **`llama3.2:3b`** (`3.2B Q4_K_M`) e **`lfm3:1.7b-q4_K_M`** (`1.7B Q4_K_M`) mantendo **exatamente o mesmo índice `embeddinggemma:300m`** e recuperação idêntica (`48/48` no replay offline).

**Principais conclusões empíricas:**
- **Desligamento 100% seguro:** com `modules.hybrid.enabled: false`, `Hybrid` **nunca é instanciado**, `pipeline.py` registra `status: "bypass"` (`0,002 ms` medianos) e executa exatamente `select_without_fusion` (zero impacto em `config/lab.yaml`, `uv.lock` e `pipeline.py`).
- **Round-Robin vs. RRF em 2 recuperadores (`bm25 + embedding`):** o RRF com $k \ge 10$ melhora a precisão no topo (`doc_recall_1`: `0,5833` $\to$ `0,6389`; `trecho_recall_1`: `0,3796` $\to$ `0,4074`; `trecho_recall_1` em consultas semânticas: `0,0556` $\to$ `0,3056`, ganho de **5,5×**), mas sofre **Fusion Tax** na janela `top_k = 5` (`trecho_recall_5` cai de `0,9630` para `0,8889`), expulsando trechos semânticos isolados no `embedding` (caso extremo em `S02`: rank `#2` no `embedding` $\to$ `#8` no RRF com $k \ge 10$).
- **Viés de ordem do Round-Robin vs. Viés lexical 2:1 do RRF:** no replay offline, inverter `rr(bm25->embedding)` para `rr(embedding->bm25)` eleva `trecho_recall_3` de `0,7593` para `0,8981` e `doc_mrr` de `0,8750` para `0,9352`. Já o RRF clássico com 3 recuperadores (`tfidf+bm25+embedding`, $k \ge 10$) colapsa `trecho_recall_5` para `0,7685` por viés lexical `2:1` (Jaccard top-12 `tfidf` vs. `bm25` = **`0,6506`**), sendo resgatado para `0,9074` por `weighted_rrf` (`1:1:2`) e superado por `convex` em `bm25+embedding` (`trecho_mrr = 0,8889`, `trecho_ndcg_5 = 0,8301`).
- **Comparação pareada de LLM no mesmo índice `embeddinggemma:300m`:** `lfm3:1.7b-q4_K_M` superou `llama3.2:3b` em precisão das citações (`0,9583`/`0,8898` vs. `0,7037`/`0,6130` por documento), porém apresentou pior abstenção em `sem_resposta` (`0,6667`/`0,8333` vs. `1,0000`) e latência `2,0×–2,8×` maior (`10,54 s`/`9,94 s` vs. `5,27 s`/`3,56 s`).

---

## 1. Fundamentação Teórica e Problemáticas do Hybrid RAG

A fusão híbrida combina listas de recuperação com naturezas estatísticas incompatíveis: escores esparsos não limitados (BM25 $\in (0, +\infty)$), cossenos normalizados TF-IDF ($\in (0, 1]$) e similaridades densas em cosseno ($\in [-1, 1]$ via operador `<=>` do `pgvector`). Na literatura de Recuperação de Informação, há duas famílias clássicas com trade-offs estruturais distintos:

### 1.1 Reciprocal Rank Fusion — RRF (Cormack, Clarke & Büttcher, SIGIR 2009)

Proposto por Cormack, Clarke e Büttcher (*SIGIR 2009*), o **RRF** descarta a magnitude bruta dos escores e agrega apenas as posições $r_j(d) \ge 1$ em cada recuperador $j \in R$:

$$\text{RRF}(d) = \sum_{j \in R_d} \frac{1}{k + r_j(d)}$$

onde $R_d \subseteq R$ é o conjunto de listas nas quais $d$ aparece (até o limite `candidate_k = C`) e $k = \text{rrf\_k} \ge 1$ é a constante de amortecimento.

**Propriedade crítica em listas curtas ($C = 12$ no `rag-lab`):** Cormack et al. (2009) calibraram $k = 60$ para listas extensas (~1.000 candidatos). Em listas curtas de comprimento $C$, um item presente em $m$ listas (mesmo em último lugar, $r = C$) supera **qualquer** item presente em $m - 1$ listas (mesmo em $r = 1$) sempre que:

$$\frac{m}{k + C} > \frac{m - 1}{k + 1} \iff k > m(C - 1) - C$$

Para $C = 12$, obtêm-se três **regimes matemáticos exatos**:

| Regime de `rrf_k` ($C = 12$) | Limiar analítico | Comportamento resultante |
|---|---|---|
| $1 \le k \le 10$ | $\frac{1}{k+1} \ge \frac{2}{k+12}$ | O **1º colocado isolado** ($m=1, r=1$) ainda empata ($k=10$: $2/22 = 1/11$) ou vence o último colocado em consenso duplo ($m=2, r=12$). |
| $10 < k \le 21$ | $\frac{2}{k+12} > \frac{1}{k+1}$ e $\frac{2}{k+1} \ge \frac{3}{k+12}$ | **Dominância cardinal parcial:** $m=2$ domina qualquer $m=1$, mas o 1º em duas listas ainda pode vencer $m=3$ na cauda. |
| $k > 21$ (inclui $k = 60, 120$) | $\frac{m}{k+12} > \frac{m-1}{k+1}$ para todo $m \le 3$ | **Dominância cardinal total ($m=3 > m=2 > m=1$):** o RRF degenera em ordenação lexicográfica `(cardinalidade m, soma recíproca)`. Um candidato fraco em `(11, 12)` vale `0,0280 > 0,0164` do 1º colocado isolado. |

Ademais, com $k = 60$ e $r \in [1, 12]$, a razão máximo/mínimo dentro de uma lista é apenas $(60+12)/(60+1) = 72/61 \approx 1{,}18$, contra $6{,}50$ em $k = 1$.

### 1.2 Combinação Convexa / Min-Max (Bruch et al., ACM TOIS 2023) e DBSF

Bruch, Gai e Ingber (*An Analysis of Fusion Functions for Hybrid Retrieval*, *ACM TOIS 2023*) demonstraram que a combinação convexa de escores normalizados supera o RRF em regimes com diferenças de magnitude informativas entre candidatos:

- **Convex (Min-Max normalizado por soma de pesos $W = \sum_{j \in R} w_j$):**

  $$\hat{s}_j^{\text{convex}}(d) = \frac{s_j(d) - s_{j,\min}}{s_{j,\max} - s_{j,\min}}, \qquad \text{Score}(d) = \sum_{j \in R_d} \frac{w_j}{W} \hat{s}_j^{\text{convex}}(d)$$

  Preserva distância relativa e é invariante a transformações lineares positivas $a s + b$, porém sofre sensibilidade a *outliers* extremos (um BM25 isolado de `25` comprime todos os demais para próximo de zero) e mapeia o último candidato ($r = C$) para `0,0`, perdendo o sinal de presença múltipla.

- **DBSF (*Distribution-Based Score Fusion*, $\mu \pm 3\sigma$ com *clip* $[0, 1]$):**

  $$\hat{s}_j^{\text{dbsf}}(d) = \min\!\left(1{,}0,\; \max\!\left(0{,}0,\; \frac{s_j(d) - (\mu_j - 3\sigma_j)}{6\sigma_j}\right)\right)$$

  Pela **desigualdade de Samuelson**, para uma amostra de tamanho $n$ com desvio padrão populacional $\sigma_j$, $|z_i| = |s_i - \mu_j|/\sigma_j \le \sqrt{n - 1}$. Logo, para $n \le 10$, $|z| \le 3$ sempre (o *clip* nunca corta); para $n = 12$, $|z| \le \sqrt{11} \approx 3{,}317$, recortando apenas *outliers* extremos e atribuindo aos candidatos finais valor $\approx 0{,}2\text{--}0{,}3$ (preservando bônus de presença tipo CombMNZ sem destruir escalas moderadas).

### 1.3 Problemáticas Estruturais Endereçadas pelo G7

1. **Viés de ordem do *Round-Robin* (`select_without_fusion`):** `src/rag_lab/pipeline.py` intercala posição a posição seguindo `MODULE_ORDER` (`tfidf -> bm25 -> embedding`). Equivalente a ordenar por `(menor_rank, índice_do_recuperador)`, ignora concordância entre métodos e favorece o recuperador listado primeiro sempre que `top_k` é ímpar (`top_k = 5`).
2. **Perda de magnitude e *Fusion Tax*:** quando um único recuperador (tipicamente `embedding` em paráfrases) detecta com alta confiança o trecho-alvo em rank `#1` ou `#2`, mas o lexical falha (`rank = null`), o RRF rebaixa esse trecho abaixo de qualquer distrator mediano encontrado nas duas listas ($m = 2$). Quando isso empurra o trecho para fora de `top_k = 5`, ocorre **Fusion Tax** com dano direto ao contexto do LLM.
3. **Viés lexical `2:1` com 3 recuperadores:** `tfidf` e `bm25` compartilham exatamente a mesma tokenização (`\w+` minúsculo sem remoção de stopwords), produzindo alta redundância. No RRF clássico com $k > 10$, qualquer trecho encontrado por ambos ($m = 2$) supera todo trecho encontrado exclusivamente pelo `embedding` ($m = 1$).
4. **Empates artificiais e `boundary_tie`:** listas baseadas apenas em rank geram empates numéricos exatos com alta frequência; no corte `evidence[:top_k]`, empates exatos na fronteira entre as posições `top_k` e `top_k + 1` são arbitrados por ordem lexicográfica do `id` (hash SHA-256).

---

## 2. Arquitetura Implementada (`hybrid.py` e `config.py`)

O núcleo do G7 em [`src/rag_lab/modules/hybrid.py`](../src/rag_lab/modules/hybrid.py) (267 linhas) e a extensão de contrato em [`src/rag_lab/config.py`](../src/rag_lab/config.py) (150 linhas) foram projetados sob **compatibilidade reversa estrita**: se a seção `hybrid:` não existir no YAML (caso do [`config/lab.yaml`](../config/lab.yaml)), o comportamento padrão permanece `rrf` clássico.

### 2.1 Contrato de Configuração e Precedência (`_resolve_hybrid_config`)

- Em [`src/rag_lab/config.py`](../src/rag_lab/config.py#L39-L119):
  - `FusionStrategy = Literal["rrf", "weighted_rrf", "convex", "dbsf"]`
  - `Weight = Annotated[StrictFloat | StrictInt, Field(ge=0, le=100, allow_inf_nan=False), AfterValidator(float)]` — bloqueia coerções silenciosas de `bool`/`str` (`isinstance(True, int) == True` em Python) e rejeita `.nan`/`.inf` já na carga do YAML.
  - `HybridConfig(StrictModel)` com `strategy: FusionStrategy = "rrf"` e `weights: dict[RetrieverName, Weight] = {}`.
  - **Validação cruzada condicionada:** todas as checagens de `hybrid` em `Settings.validate_pipeline` só executam **se `self.modules["hybrid"].enabled` for `True`**, evitando quebrar testes ou modos que desligam o módulo mantendo a seção `hybrid:` no YAML.
- Em [`src/rag_lab/modules/hybrid.py`](../src/rag_lab/modules/hybrid.py#L96-L133), `_resolve_hybrid_config(payload, settings)` respeita a precedência:
  1. `settings.hybrid` (quando declarado via `config.py`);
  2. fallback autocontido em `payload["attributes"]["hybrid_config"]` (validação completa em runtime para ambientes de arquivo único onde `config.py` não foi alterado);
  3. padrão `("rrf", {})`.

### 2.2 Padrão *Strategy*, Precisão Numérica e Invariantes

```text
payload["retrieval"] (imutável)
   ├── _extract_raw_scores() ──> rejeita bool / NaN / Inf (ValueError explícito)
   ├── _resolve_weights()    ──> valida cobertura exata das chaves e sum(weights) > 0
   ├── STRATEGIES[strategy]  ──> rrf | weighted_rrf | convex | dbsf (contribuições pré-escaladas)
   ├── math.fsum(contribs)   ──> elimina discrepância de 1 ULP independente da ordem das chaves
   └── sorted((-score, id))  ──> desempate determinístico invariante à escala bruta
        ├── item["score"] = fused e item["scores"]["hybrid"] = fused (sem chaves espúrias)
        ├── preserva metadata["tags"] e atributos do usuário; NUNCA trunca para top_k
        └── grava payload["attributes"]["hybrid"] (schema "g7-hybrid/1", stats, overlap, boundary_tie, ranking)
```

Destaques técnicos verificados na suíte [`tests/test_hybrid.py`](../tests/test_hybrid.py):
- **Soma exata com `math.fsum`:** em `rrf_k = 60`, a soma padrão IEEE-754 de ranks `{1, 2, 7}` em ordens distintas difere por 1 ULP (`((1/61 + 1/67) + 1/62) < ((1/62 + 1/67) + 1/61)`). `math.fsum` garante arredondamento exato e fechamento bit a bit `math.fsum(contributions.values()) == fused`.
- **Proteção contra degenerescências e `NaN`:** em `convex` (`span <= 0.0`) e `dbsf` (`std <= 0.0`), incluindo listas unitárias (`len == 1`), a contribuição retorna `weight / weight_sum` em vez de dividir por zero; listas vazias retornam `[]` e estatísticas `null` na telemetria.
- **Preservação integral de contrato:** `metadata["tags"]` permanece inalterado via `deepcopy` na primeira ocorrência do chunk; `item["scores"]` recebe exclusivamente a chave `"hybrid"`; `payload["evidence"]` mantém toda a união ranqueada para consumo de etapas a jusante (`reranking`, `sentence`, `build_context`).

---

## 3. Garantia de Desligamento Seguro (`hybrid: false`) e Testes de Regressão

Na revisão em aula do PR #6, o professor levantou a hipótese crítica de que desabilitar `hybrid` (`enabled: false`) alteraria o comportamento original do laboratório. Uma **auditoria forense linha a linha** foi executada antes das coletas:

1. **Integridade do núcleo de orquestração:** `git diff upstream/develop -- src/rag_lab/pipeline.py config/lab.yaml uv.lock` retornou **vazio** (`0` linhas modificadas).
2. **Mapeamento exato do fluxo em [`src/rag_lab/pipeline.py`](../src/rag_lab/pipeline.py#L95-L112):**
   - Na iteração `name == "hybrid"`, `select_without_fusion(payload)` é executado antes do switch `enabled = settings.modules["hybrid"].enabled`.
   - Quando `enabled` é `False`, o objeto `Hybrid` **não é instanciado**, `trace` registra `{"module": "hybrid", "status": "bypass"}` e `payload["evidence"]` preserva exatamente a saída *round-robin* de `select_without_fusion`.
3. **Verificação em 7 cenários controlados + suíte [`tests/test_g7_regression.py`](../tests/test_g7_regression.py):**
   - `config/lab.yaml` intacto (`embedding: true`, demais 9 módulos `false`), `hybrid` nunca instanciado (`monkeypatch` em `REGISTRY["hybrid"]` falha se chamado), `metadata["tags"]` preservadas.
   - `bm25 + embedding` (`2` recuperadores) e `tfidf + bm25 + embedding` (`3` recuperadores) com `hybrid: false`: sequência de IDs de `result.evidence` idêntica à `select_without_fusion(payload)`.
   - YAML declarando bloco `hybrid: {strategy: dbsf, weights: ...}` ou `weighted_rrf` com `modules.hybrid.enabled: false`, bem como requisições passando `attributes.hybrid_config`: completamente ignorados em modo `bypass`.
4. **Causas operacionais identificadas para o sintoma observado em aula:**
   - **Recarga em `lifespan` na API ([`src/rag_lab/api.py`](../src/rag_lab/api.py#L17-L22)):** `load_settings(config_path)` roda uma única vez na subida do servidor FastAPI; alterar o YAML sem reiniciar o container `app` mantém `hybrid: true` ativo em memória.
   - **Precedência de `.env` (`RAG_CONFIG`):** quando `.env` define `RAG_CONFIG=config/lab.local.yaml`, edições em `config/lab.yaml` são ignoradas pela API/CLI.
   - **Diferença entre 1 e 2+ recuperadores ativos:** desligar apenas `hybrid` mantendo `bm25: true` e `embedding: true` aciona `select_without_fusion` (*round-robin*), cuja ordenação difere de `embedding` isolado (`config/lab.yaml`).
5. **Comprovação empírica na bateria online (`t4_latencia_chamadas.md`):** nas configurações `e1_rr_bm25_emb` e `e1_rr_todos` (`hybrid.enabled: false`), o pipeline executou em `hybrid_status = bypass` com mediana de `0,002 ms`, `0` casos em `t5_fusion_tax.md` e IDs 100% compatíveis no `replay_verificacao.json`.

---

## 4. Protocolo Experimental Controlado

Para cumprir o controle rígido de variáveis e eliminar confusão entre modelos, toda a coleta em [`experiments/g7/`](../experiments/g7/README.md) obedeceu às seguintes condições:

- **Corpus autoral sob licença CC-BY-4.0 (`experiments/g7/corpus/`):** `12` documentos `.md` didáticos sobre o sistema fictício "Plataforma Núcleo Aurora" (`23.991` caracteres, segmentados deterministicamente em **`135` chunks** sentenciais), desenhados para impedir memória prévia do LLM e tensionar códigos literais, paráfrases semânticas, fatos multi-documento e armadilhas lexicais ("token", "rotação", "expurgo", "janela").
- **Gabarito independente (`experiments/g7/perguntas_gabarito.yaml`):** `24` perguntas (`6` `literal` `L01–L06`, `6` `semantica` `S01–S06`, `6` `composta` `C01–C06`, `6` `sem_resposta` `N01–N06` sendo 3 *near-miss* + 3 fora de domínio), com `35` trechos relevantes verificados 1:1 dentro de chunk único.
  - **`gabarito_sha256` (`hash_mode: sha256-lf`):** `046841378bc39d6dd95fd2edefe37f3d91161afed85802925d6e932c68762e32`.
  - **Transparência metodológica:** os `24` itens do gabarito ainda mantêm `status_revisao: "pendente_validacao_humana"` (`gabarito_homologado: false` no manifesto) porque o arquivo não foi alterado durante este ciclo de auditoria e coleta.
- **Banco dedicado e ingestão única:** banco `rag_g7` (`pgvector/pgvector:pg16`), ingerido uma única vez em `20,72 s` (`12` docs / `135` chunks, `RAG_TAG_CLASSIFIER=terms`) com **`embeddinggemma:300m`** (`307.58M BF16`, `768` dimensões, digest `85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1`).
- **Parâmetros constantes em todas as rodadas:** `top_k = 5`, `candidate_k = 12`, `max_context_chars = 12000`, `llm.temperature = 0.0`, `multi_query = false`, `hyde = false`, `decomposicao = false`, `reranking = false`, `sentence = false`, `compressao = false`.
- **Duas baterias controladas executadas com `experiments/g7/rodar.py` e avaliadas com `experiments/g7/avaliar.py`:**

| Bateria | Diretório de Saída / Resultados | Modelos (com Digests Reais Ollama) | Escopo Executado | Verificação de Replay |
|---|---|---|---|---|
| **Bateria A — Principal (`rr_vs_rrf`)** | [`experiments/g7/results/rr_vs_rrf/`](../experiments/g7/results/rr_vs_rrf/t1_recuperacao.md) | LLM: **`llama3.2:3b`** (`3.2B Q4_K_M`, `a80c4f17acd55265...b8b72`) · Emb: **`embeddinggemma:300m`** (`85462619ee721b46...4679f1`) | `6` rodadas × `24` perguntas = **`144` consultas** (`e1_rr_bm25_emb`, `e1_rr_todos`, `e2_rrf_k1`, `k10`, `k60`, `k120`) | `total_checked: 144`, `all_matched: true` |
| **Bateria B — Referência Pareada de LLM (`llm_ref_gemma`)** | [`experiments/g7/results/llm_ref_gemma/`](../experiments/g7/results/llm_ref_gemma/t1_recuperacao.md) | LLM: **`lfm3:1.7b-q4_K_M`** (`1.7B Q4_K_M`, `b36a103fe2f8fbe9...570b5d`) · Emb: **`embeddinggemma:300m`** (mesmo índice) | `2` rodadas × `24` perguntas = **`48` consultas** (`e1_rr_bm25_emb`, `e2_rrf_k60_bm25_emb`) | `total_checked: 48`, `all_matched: true` |

> **Simetria de avaliação (`avaliar.py`):** enquanto `Hit@5`, `Recall@5` e `nDCG@5` medem exatamente os 5 chunks entregues ao LLM por `build_context`, `Recall@10` e `MRR` reconstroem a lista completa pós-seleção (`select_without_fusion` em `E1`, `attributes.hybrid.ranking` em `E2`/`E4`) para não penalizar configurações sem `attributes.hybrid`.

---

## 5. Resultados Controlados (`T1–T6` e Comparação Pareada de LLM)

### 5.1 T1 — Recuperação Online (`llama3.2:3b` + `embeddinggemma:300m`, 18 respondíveis)

| Configuração | Perfil | `doc_rec@1` | `doc_rec@5` | `doc_mrr` | `trecho_rec@1` | `trecho_rec@3` | `trecho_rec@5` | `trecho_rec@10` | `trecho_hit@5` | `trecho_mrr` | `trecho_ndcg@5` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`e1_rr_bm25_emb`** (`bypass`) | `TODOS` (`18`) | `0.5833` | **`1.0000`** | `0.8750` | `0.3796` | `0.7593` | **`0.9630`** | `1.0000` | **`1.0000`** | **`0.8333`** | **`0.8149`** |
| `e1_rr_bm25_emb` | `literal` (`6`) | `1.0000` | `1.0000` | `1.0000` | **`0.6667`** | `1.0000` | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** | **`0.9732`** |
| `e1_rr_bm25_emb` | `semantica` (`6`) | `0.2500` | `1.0000` | `0.6250` | `0.0556` | `0.3889` | **`0.9444`** | `1.0000` | `1.0000` | `0.5000` | **`0.5430`** |
| `e1_rr_bm25_emb` | `composta` (`6`) | `0.5000` | `1.0000` | `1.0000` | `0.4167` | `0.8889` | `0.9444` | `1.0000` | `1.0000` | `1.0000` | `0.9284` |
| **`e1_rr_todos`** (`bypass`, 3 ret.) | `TODOS` (`18`) | `0.5833` | `0.9722` | `0.8657` | `0.3796` | `0.7593` | `0.9444` | `0.9815` | `1.0000` | `0.8185` | `0.7921` |
| **`e2_rrf_k1_bm25_emb`** (`k=1`) | `TODOS` (`18`) | `0.5833` | `0.9167` | `0.8704` | `0.3796` | **`0.8056`** | `0.8889` | `1.0000` | `0.9444` | `0.8009` | `0.7802` |
| **`e2_rrf_k10/60/120_bm25_emb`** | `TODOS` (`18`) | **`0.6389`** | `0.9722` | **`0.8843`** | **`0.4074`** | `0.7778` | `0.8889` | `1.0000` | `0.9444` | `0.8144` | `0.7815` |
| `e2_rrf_k10/60/120_bm25_emb` | `literal` (`6`) | `1.0000` | `1.0000` | `1.0000` | `0.5000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `0.8333` | `0.8978` |
| `e2_rrf_k10/60/120_bm25_emb` | `semantica` (`6`) | **`0.4167`** | `0.9167` | **`0.6528`** | **`0.3056`** | `0.3889` | `0.7222` | `1.0000` | `0.8333` | **`0.6097`** | `0.5095` |
| `e2_rrf_k10/60/120_bm25_emb` | `composta` (`6`) | `0.5000` | `1.0000` | `1.0000` | `0.4167` | **`0.9444`** | `0.9444` | `1.0000` | `1.0000` | `1.0000` | **`0.9373`** |

**Leitura objetiva da Tabela T1 ([`t1_recuperacao.md`](../experiments/g7/results/rr_vs_rrf/t1_recuperacao.md)):**
1. **Platô exato para $k \ge 10$ em `bm25 + embedding`:** `e2_rrf_k10`, `e2_rrf_k60` e `e2_rrf_k120` produziram **exatamente os mesmos valores** em todas as métricas e perfis, confirmando o limiar teórico de dominância $k = 10$ para $C = 12$.
2. **RRF melhora a concentração no topo (Rank 1 e consultas compostas):** no geral, `doc_recall_1` sobe de `0,5833` para `0,6389`, `doc_mrr` sobe para `0,8843` e `trecho_recall_1` sobe de `0,3796` para `0,4074` — puxado pelas consultas semânticas (`trecho_recall_1` de `0,0556` para `0,3056` e `trecho_mrr` de `0,5000` para `0,6097`, porque o `rr(bm25->embedding)` sempre colocava o 1º candidato do BM25 em Rank 1) e compostas (`trecho_recall_3` de `0,8889` para `0,9444` e `trecho_ndcg_5` para `0,9373`).
3. **Round-Robin vence em cobertura de janela `top_k = 5` (`trecho_recall_5` e `trecho_mrr` global):** por reservar cota alternada para `bm25` e `embedding`, `e1_rr_bm25_emb` preserva `trecho_hit_5 = 1,0000` e `trecho_recall_5 = 0,9630` (vs. `0,8889` do RRF), evitando a expulsão de evidências semânticas sem suporte lexical. Já `e1_rr_todos` (`tfidf+bm25+embedding`) perde cobertura (`trecho_recall_5 = 0,9444`) porque dois recuperadores lexicais consomem `4` das `5` primeiras vagas do `top_k = 5`.

### 5.2 T2 — Replay Offline Exato: Curva `rrf_k`, Viés do Round-Robin e Estratégias `E4`

O motor de *replay* offline sobre `retrieval` (`100%` idêntico ao online nos `144` envelopes de A e `48` de B) revelou três resultados centrais em [`t2_curva_rrf_k.md`](../experiments/g7/results/rr_vs_rrf/t2_curva_rrf_k.md):

1. **Curva `rrf_k` ($\{1, 3, 5, 10, 11, 15, 21, 22, 30, 60, 120, 1000\}$):**
   - Em **`bm25+embedding`**: `rrf_k = 3` atinge o melhor equilíbrio de RRF (`doc_mrr = 0,8907`, `trecho_mrr = 0,8246`, `trecho_ndcg_5 = 0,7871`, menor `boundary_tie_rate = 0,0556`). De `k = 10` até `k = 1000`, todas as métricas entram no platô `trecho_mrr = 0,8144`, `trecho_recall_5 = 0,8889` e `boundary_tie_rate = 0,1667`.
   - Em **`tfidf+bm25+embedding`** (3 recuperadores): o RRF sofre **degradação severa de cobertura** quando $k \ge 10$: `trecho_recall_5` despenca de `0,8241` (`k=1`) para **`0,7685`**, e `trecho_recall_10` despenca de **`0,9815`** (`k=1, 3`) para **`0,8241`** (`k >= 10`), porque candidatos irrelevantes encontrados por `tfidf` + `bm25` ($m=2$) enterram até mesmo além da 10ª posição trechos semânticos encontrados só pelo `embedding` ($m=1$).

2. **Prova do viés posicional do Round-Robin (`E3_permutacao_round_robin`):**

| Permutação Round-Robin (`select_without_fusion`) | `doc_mrr` | `doc_ndcg_5` | `trecho_recall_3` | `trecho_recall_5` | `trecho_mrr` | `trecho_ndcg_5` |
|---|---:|---:|---:|---:|---:|---:|
| `rr(bm25->embedding)` *(padrão atual `MODULE_ORDER`)* | `0.8750` | `0.8976` | `0.7593` | `0.9630` | `0.8333` | `0.8149` |
| `rr(embedding->bm25)` *(ordem invertida)* | **`0.9352`** | **`0.9360`** | **`0.8981`** | `0.9630` | **`0.8426`** | **`0.8363`** |
| `rr(tfidf->bm25->embedding)` *(padrão atual 3 ret.)* | `0.8657` | `0.8776` | `0.7593` | `0.9444` | `0.8185` | `0.7921` |
| `rr(embedding->bm25->tfidf)` *(denso primeiro, 3 ret.)* | **`0.9306`** | **`0.9298`** | **`0.8148`** | **`0.9630`** | **`0.8380`** | **`0.8310`** |

Mudar **exclusivamente** a ordem do dicionário no *round-robin* (sem fundir escores) altera `+0,1388` em `trecho_recall_3` (`bm25+embedding`) e `+0,0649` em `doc_mrr` (`3` recuperadores), demonstrando que a qualidade do *round-robin* depende diretamente de um viés estrutural arbitrário herdado de `MODULE_ORDER`.

3. **Comparação de estratégias `E4` sobre exatamente os mesmos candidatos:**

| Recuperadores | Estratégia | `doc_recall_5` | `doc_mrr` | `trecho_recall_3` | `trecho_recall_5` | `trecho_recall_10` | `trecho_mrr` | `trecho_ndcg_5` | `boundary_tie` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bm25+embedding` | `rrf` (`k=60`) | `0.9722` | `0.8843` | `0.7778` | `0.8889` | `1.0000` | `0.8144` | `0.7815` | `0.1667` |
| `bm25+embedding` | `weighted_rrf` (`1:2`, `k=60`) | **`1.0000`** | `0.8843` | `0.7500` | **`0.9259`** | `1.0000` | `0.8194` | `0.8040` | **`0.0000`** |
| `bm25+embedding` | **`convex`** (`1:1`) | `0.9167` | **`0.8981`** | **`0.8426`** | `0.9074` | `0.9815` | **`0.8889`** | **`0.8301`** | **`0.0000`** |
| `bm25+embedding` | `dbsf` (`1:1`) | `0.9722` | `0.8889` | `0.7778` | `0.9074` | `0.9815` | `0.8514` | `0.8106` | **`0.0000`** |
| `tfidf+bm25+emb` | `rrf` (`k=60`) | `0.9444` | `0.8935` | `0.7500` | `0.7685` | `0.8241` | `0.8214` | `0.7432` | `0.0000` |
| `tfidf+bm25+emb` | **`weighted_rrf`** (`1:1:2`, `k=60`) | **`0.9722`** | `0.9213` | `0.7500` | **`0.9074`** | **`1.0000`** | `0.8144` | `0.7878` | `0.0000` |
| `tfidf+bm25+emb` | **`convex`** (`1:1:1`) | **`0.9722`** | `0.9074` | `0.7315` | `0.8796` | `0.9815` | **`0.8413`** | **`0.7917`** | `0.0000` |
| `tfidf+bm25+emb` | `dbsf` (`1:1:1`) | `0.9444` | **`0.9352`** | `0.7315` | `0.7685` | `0.8519` | `0.8249` | `0.7414` | `0.0000` |

Em `bm25+embedding`, **`convex`** lidera em `trecho_mrr` (`0,8889` vs. `0,8144` do RRF e `0,8333` do Round-Robin) e `trecho_ndcg_5` (`0,8301`), além de zerar empates na fronteira (`boundary_tie_rate = 0,0`). Com `3` recuperadores, **`weighted_rrf` (`1:1:2`)** corrige o viés lexical `2:1` e resgata `trecho_recall_5` de `0,7685` para `0,9074` (`+0,1389`).

### 5.3 T5 e T6 — Casos de *Fusion Tax*, Redundância (Jaccard Top-12) e Empates na Fronteira

Foram detectados **`29` casos de *Fusion Tax*** nas 4 rodadas RRF online da Bateria A ([`t5_fusion_tax.md`](../experiments/g7/results/rr_vs_rrf/t5_fusion_tax.md)): `7` casos idênticos em cada platô ($k \in \{10, 60, 120\}$) e `8` casos em `k = 1` (`S05` entra como `queda_mrr_no_top5`). Os **`3` casos severos (`perda_top5`)** que ocorrem em **todas** as variantes RRF são:

1. **`S02` (`semantica`, `06_seguranca_criptografia.md`):** o `embedding` encontra o trecho-alvo em **rank `#2`**, enquanto `bm25` não o coloca entre os 12 candidatos (`null`). No *round-robin*, entra em **`#4`** (`top_k = 5`, `Recall@5 = 1,0`). No RRF com $k \in \{10, 60, 120\}$, cai para **rank `#8`** (`fused_recall_5 = 0,0`, `delta_recall5 = -1,0`, `delta_mrr = -0,375`), expulso do contexto do LLM; mesmo em `k = 1` cai para `#6` (ainda fora do `top_k = 5`).
2. **`S01` (`semantica`):** `trecho_recall_5` cai de `0,6667` (isolado/`rr`) para **`0,3333`** no RRF (`delta_recall5 = -0,3333`).
3. **`C03` (`composta`):** `trecho_recall_5` cai de `1,0000` (`embedding` isolado) para **`0,6667`** (`delta_recall5 = -0,3333`).

Em [`t6_sobreposicao_empates.md`](../experiments/g7/results/rr_vs_rrf/t6_sobreposicao_empates.md):
- **Complementaridade `bm25` vs. `embedding`:** Jaccard top-12 médio de apenas **`0,2091`** (`3,958` itens comuns em média, variando de `0,1217` em `sem_resposta` e `0,1437` em `semantica` a `0,3571` em `composta`).
- **Redundância lexical `tfidf` vs. `bm25` (`e1_rr_todos`):** Jaccard top-12 médio de **`0,6506`** (chegando a `0,7255` em `composta` e `0,7011` em `literal`), comprovando matematicamente por que ambos votam juntos no RRF de 3 recuperadores.
- **Empates na fronteira `top_k = 5`:** `boundary_tie_rate = 0,25` (`6/24` consultas em todas as execuções RRF, chegando a `0,50` em `sem_resposta` e `0,3333` em `semantica` para $k \ge 10$).

### 5.4 T3 e T4 — Geração, Citações, Abstenção e Latência (`rr_vs_rrf`)

- **Geração com `llama3.2:3b` ([`t3_geracao_citacoes.md`](../experiments/g7/results/rr_vs_rrf/t3_geracao_citacoes.md)):**
  - **Abstenção em `N01–N06`:** **`1,0000` (`6/6`) em todas as 6 rodadas** — nenhuma alucinação detectada pela heurística de recusa.
  - **Validade sintática de citações `[n]`:** `1,0000` nas 5 rodadas `bm25+embedding` (`0,9444` em `e1_rr_todos`).
  - **Precisão das fontes citadas frente ao gabarito (`precisao_fontes_citadas_trecho`):** `0,7241` (`e1_rr_todos`) e `0,6435` (`e1_rr_bm25_emb`) superam `e2_rrf_k10` (`0,6222`), `e2_rrf_k1` (`0,5861`) e `e2_rrf_k60/k120` (`0,5630`), refletindo a perda de `S02` e `S01` na janela `top_k = 5`. A avaliação humana final (rubrica 0–2) encontra-se exportada em [`avaliacao_humana_template.csv`](../experiments/g7/results/rr_vs_rrf/avaliacao_humana_template.csv).
- **Latência e chamadas ([`t4_latencia_chamadas.md`](../experiments/g7/results/rr_vs_rrf/t4_latencia_chamadas.md)):**
  - Custo de rede constante: exatas **`24` `chat_calls`**, **`24` `embed_calls`** e **`24` `embed_texts`** por rodada.
  - `hybrid_ms_mediana`: `0,002 ms` em `bypass` (`e1_*`) e **`0,909–1,603 ms`** (`p95 <= 1,976 ms`) com RRF ativo — **menos de `0,05%` do `wall_time_ms_mediana`** (`3.658,8–5.461,0 ms`), dominado em mais de `97%` pelo tempo de geração do LLM (`3.559,4–5.360,7 ms`).

### 5.5 Comparação Pareada de LLM sobre o Mesmo Embedding (`embeddinggemma:300m`)

Na Bateria B ([`experiments/g7/results/llm_ref_gemma/`](../experiments/g7/results/llm_ref_gemma/t1_recuperacao.md)), substituiu-se **apenas** o gerador para `lfm3:1.7b-q4_K_M`, preservando intactos o banco `rag_g7`, o índice `embeddinggemma:300m`, os recuperadores e os parâmetros de pipeline. Conforme comprovado por [`t1_recuperacao.md`](../experiments/g7/results/llm_ref_gemma/t1_recuperacao.md), **todas as métricas de recuperação (`T1`), *Fusion Tax* (`T5`) e Jaccard/empates (`T6`) foram 100% idênticas** às da Bateria A (`rr_vs_rrf`), isolando perfeitamente o efeito puro do modelo gerador:

| Configuração | LLM (sobre `embeddinggemma:300m`) | `taxa_respostas_com_citacao` | `precisao_cit_doc` | `precisao_cit_trecho` | `cobertura_fatos_heuristica` | `abstencao_N01_N06` | `generation_ms_mediana` | `wall_time_ms_p95` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `e1_rr_bm25_emb` | `llama3.2:3b` (`3.2B Q4_K_M`) | `0.8333` | `0.7037` | `0.6435` | `0.6707` | **`1.0000`** (`6/6`) | **`5272.57 ms`** | **`8551.51 ms`** |
| `e1_rr_bm25_emb` | `lfm3:1.7b-q4_K_M` (`1.7B Q4_K_M`) | **`1.0000`** | **`0.9583`** | **`0.8556`** | **`0.7680`** | `0.6667` (`4/6`) | `10535.45 ms` | `24508.49 ms` |
| `e2_rrf_k60_bm25_emb` | `llama3.2:3b` (`3.2B Q4_K_M`) | `0.7778` | `0.6130` | `0.5630` | `0.6625` | **`1.0000`** (`6/6`) | **`3559.39 ms`** | **`10101.89 ms`** |
| `e2_rrf_k60_bm25_emb` | `lfm3:1.7b-q4_K_M` (`1.7B Q4_K_M`) | **`1.0000`** | **`0.8898`** | **`0.8130`** | **`0.7140`** | `0.8333` (`5/6`) | `9942.65 ms` | `25484.80 ms` |

**Diagnóstico comparativo pareado:**
1. **Efeito consistente da recuperação na geração:** nos **dois** LLMs, trocar *round-robin* por `rrf(k=60)` reduz a precisão das fontes citadas por trecho (`-0,0805` no `llama3.2:3b` e `-0,0426` no `lfm3:1.7b-q4_K_M`), consequência direta do rebaixamento de `S02` e `S01` para fora de `top_k = 5`.
2. **Trade-off aderência a citações vs. abstenção e latência:** `lfm3:1.7b-q4_K_M` é mais aderente ao formato de citações `[n]` (`100%` com citação, `0,9583` de precisão documental em `e1_rr`) e cobre mais fatos esperados (`0,7680` vs. `0,6707`), porém falha na abstenção em parte dos *near-miss* (`4/6` em RR e `5/6` em RRF vs. `6/6` do `llama3.2:3b`) e gera respostas bem mais longas (`p95 ~ 25 s`).

---

## 6. Limitações Técnicas e Experimentais

1. **Perda de magnitude bruta no RRF:** por depender exclusivamente do posto inteiro $r_j(d)$, o RRF trata como equivalente uma liderança esmagadora (`BM25 = 24,0` vs. `2º = 3,0`) e um empate quase nulo; com `candidate_k = 12` e $k \ge 11$, transforma-se em votação de cardinalidade ($m$).
2. **Fusion Tax em consultas puramente semânticas (`S02`, `S01`, `C03`):** quando uma paráfrase não compartilha termos lexicais com o chunk-alvo, candidatos medianos encontrados por dois métodos ou distraidores lexicais deslocam o 1º/2º lugar denso para além do corte `top_k = 5`.
3. **Empates na fronteira (`boundary_tie = 25%`):** em `6` das `24` consultas com RRF `1:1`, o 5º e o 6º colocados empatam exatamente no score recíproco, fazendo o ingresso no contexto do LLM depender do hash SHA-256 (`id`) do chunk em vez da força semântica ou lexical.
4. **Viés lexical `2:1` quando `tfidf`, `bm25` e `embedding` operam juntos:** com Jaccard médio de `0,6506` entre `tfidf` e `bm25`, o RRF clássico degrada `trecho_recall_5` para `0,7685` e `trecho_recall_10` para `0,8241`.
5. **Ausência de calibração absoluta para abstenção na camada de recuperação:** todas as estratégias por rank ou normalização por consulta (`rrf`, `weighted_rrf`, `convex`, `dbsf`) produzem sempre um topo próximo de `m/(k+1)` ou `1,0`, não fornecendo sinal de corte para `N01–N06` — a abstenção depende exclusivamente do prompt e da disciplina do LLM.
6. **Acoplamento com `reranking` a jusante (`src/rag_lab/modules/reranking.py`):** como `Hybrid` não trunca a lista fundida e `Reranking` ordena toda `evidence` por `(-overlap, id)`, habilitar `reranking: true` descarta integralmente a ordenação produzida tanto pelo `round-robin` quanto pelo `hybrid` (experimentos G7 devem manter `reranking: false`).
7. **Limitações do protocolo empírico:**
   - **Tamanho amostral:** `12` documentos (`135` chunks) e `24` perguntas (`6` por perfil) limitam a generalização estatística de pesos ajustados sem risco de *overfitting*.
   - **Chunks de cabeçalho (`position = 0`):** títulos curtos em Markdown geram chunks próprios na segmentação sentencial de `ingestion.py`, podendo ocupar vagas no `top_k = 5` sem conter os fatos completos da seção.
   - **Sensibilidade ao embedding em PT-BR:** o modelo `embeddinggemma:300m` (`768` dims) responde por toda a representação densa deste estudo; outros embeddings alteram diretamente a sobreposição `Jaccard` e a incidência de *Fusion Tax*.
   - **Homologação humana pendente:** o gabarito mantém `status_revisao: "pendente_validacao_humana"` e o preenchimento manual da rubrica `0–2` de fidelidade em [`avaliacao_humana_template.csv`](../experiments/g7/results/rr_vs_rrf/avaliacao_humana_template.csv) permanece pendente.

---

## 7. Decisão Técnica e Recomendações Práticas

| Cenário no `rag-lab` | Estratégia Recomendada | Parâmetros Indicados | Justificativa Baseada nos Dados |
|---|---|---|---|
| **Baseline legado ou recuperação de máxima cobertura `top_k=5` sem `reranking`** | `hybrid: {enabled: false}` (*round-robin*) | `bm25 + embedding` (`top_k = 5`) | Garante cota alternada e preservou `trecho_recall_5 = 0,9630` e `trecho_hit_5 = 1,0000` sem *Fusion Tax* em `S02` (sujeito a viés posicional de `MODULE_ORDER`). |
| **Precisão em Rank 1 e respostas factuais/compostas com RRF clássico** | `hybrid` com `rrf` | `bm25 + embedding`, `rrf_k` calibrado (`3` a `10`) | Supera o *round-robin* em `doc_recall_1` (`0,6389`), `trecho_recall_1` (`0,4074`) e `trecho_ndcg_5` em compostas (`0,9373`); `rrf_k = 60` é alto demais para `candidate_k = 12`. |
| **3 recuperadores ativos (`tfidf + bm25 + embedding`)** | **`weighted_rrf`** *(obrigatório frente ao RRF clássico)* | `weights: {tfidf: 1.0, bm25: 1.0, embedding: 2.0}` | Corrige o viés lexical `2:1` (Jaccard `0,6506`), elevando `trecho_recall_5` de `0,7685` (RRF) para **`0,9074`** e `trecho_recall_10` de `0,8241` para **`1,0000`**. |
| **Melhor equilíbrio ranking + desempate sem hash (`bm25 + embedding`)** | **`convex`** ou **`dbsf`** | `weights: {bm25: 1.0, embedding: 1.0}` | No replay sobre `bm25+embedding`, **`convex`** atingiu os maiores **`trecho_mrr = 0,8889`** e **`trecho_ndcg_5 = 0,8301`**, enquanto `dbsf` manteve `doc_recall_5 = 0,9722` e `trecho_mrr = 0,8514`, ambos zerando `boundary_tie` (`0,0`). |
| **Escolha do gerador sobre índice `embeddinggemma:300m`** | **`llama3.2:3b`** (produção equilibrada) vs. `lfm3:1.7b-q4_K_M` (citação densa) | `temperature: 0.0` | `llama3.2:3b` absteve em `100%` (`6/6`) das perguntas sem resposta e foi ~2×–2,8× mais rápido; `lfm3` teve maior precisão de citações (`0,8556` vs. `0,6435`), porém errou 1–2 casos *near-miss*. |

---

## Referências Bibliográficas

- **Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009).** *Reciprocal rank fusion outperforms condorcet and individual rank learning methods.* Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09), pp. 758–759.
- **Bruch, S., Gai, S., & Ingber, A. (2023).** *An Analysis of Fusion Functions for Hybrid Retrieval.* ACM Transactions on Information Systems (TOIS), 42(1), Article 20.
- **Samuelson, P. A. (1968).** *How Deviant Can You Be?* Journal of the American Statistical Association, 63(324), 1522–1525 (fundamento teórico do limite $|z| \le \sqrt{n-1}$ no recorte $\mu \pm 3\sigma$ do DBSF).
