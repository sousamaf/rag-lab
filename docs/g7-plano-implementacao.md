# G7 · Hybrid — Documento Vivo de Engenharia e Planejamento

> **Documento vivo da missão G7 (`hybrid.py`).** Este arquivo registra a arquitetura, as decisões de engenharia (ADRs), a fundamentação matemática, o código já implementado com suas justificativas e o plano executável das próximas fases. Deve ser atualizado ao término de cada fase (**F0 a F11**).

---

## 1. Cabeçalho e Painel de Status Vivo

### 1.1 Metadados do Trabalho

| Campo | Valor |
|---|---|
| **Grupo** | **G7 — Hybrid** |
| **Integrantes** | Diego · Gabriel Alves · João Victor Póvoa |
| **Repositório / Branch** | `rag-lab` · branch completa `feat/g7-hybrid` (rebased sobre `upstream/develop` `923cd03`, base original `e230073`) + branch de arquivo único `g7-hybrid` |
| **Módulo principal** | `src/rag_lab/modules/hybrid.py` (com contrato YAML estendido em `src/rag_lab/config.py` e fallback autocontido em `attributes.hybrid_config`) |
| **Missão oficial (`docs/missoes.md`)** | Comparar seleção básica *round-robin* (`select_without_fusion`) e **Reciprocal Rank Fusion (RRF)** variando `rrf_k`, entregando código sólido, comparação controlada e análise honesta das limitações da técnica |
| **Ambição técnica do G7** | Evoluir `hybrid.py` com arquitetura *Strategy* retrocompatível via YAML (`rrf`, `weighted_rrf`, `convex`, `dbsf`), precisão numérica (`math.fsum`), proteção *NaN-free*, telemetria diagnóstica em `payload["attributes"]["hybrid"]` e suíte de avaliação experimental com *replay* offline exato |
| **Diagramas de referência** | [Fluxo didático](diagrams/rag-didatico.svg) · [Mapa dos módulos e grupos](diagrams/rag-modulos.svg) · [Pipeline modular](diagrams/rag-pipeline.svg) |

### 1.2 Painel de Progresso das Fases (F0 a F11)

```text
Progresso Global da Missão G7 (F0 a F11): [#####################---] ~83% (9,9 / 12 fases)
  [x] F0  Setup & Baseline            100%  [####################]
  [x] F1  Testes de Caracterização    100%  [####################]
  [x] F2  Contrato YAML HybridConfig  100%  [####################]
  [x] F3  Núcleo Strategy + RRF/fsum  100%  [####################]
  [x] F4  Estratégia Weighted RRF     100%  [####################]
  [x] F5  Estratégia Convex Min-Max   100%  [####################]
  [x] F6  Estratégia DBSF (mu+-3sig)  100%  [####################]
  [x] F7  Composição Pipeline + Sync  100%  [####################]
  [~] F8  Corpus, Gabarito & Piloto    90%  [##################--] (falta revisão humana + freeze SHA-256)
  [x] F9  Ferramental rodar/avaliar   100%  [####################]
  [ ] F10 Execução Matriz E0-E6         0%  [--------------------]
  [ ] F11 Avaliação Humana, Doc & PR    0%  [--------------------]
```

| Fase | Status | Progresso | Arquivos-alvo | Critério de pronto | Bloqueios / Próxima Ação | Data / Checkpoint |
|---|---|---|---|-|---|---|
| **F0** | `[x] Concluída` | `100%` | Ambiente `.venv` (`uv 0.13.9`, CPython `3.12.16`), branch `feat/g7-hybrid` | `uv sync --frozen --extra dev --python 3.12`, `rag-lab validate`, `rag-lab doctor --offline`, `ruff check .` e `pytest` verdes sem editar arquivos | Nenhum (concluída) | 2026-11-18 · Base `e230073` validada (31/31 testes unitários verdes no sandbox) |
| **F1** | `[x] Concluída` | `100%` | `tests/test_hybrid.py` (novo) | Testes de caracterização do RRF clássico, variação de `rrf_k`, limiar de dominância $C=12$, não-truncagem, imutabilidade de `retrieval` e viés do *round-robin* passando contra o código original | Nenhum (concluída) | 2026-11-18 · `9 passed` contra `hybrid.py` original + `ruff` limpo |
| **F2** | `[x] Concluída` | `100%` | `src/rag_lab/config.py`, `tests/test_config.py` | `HybridConfig` opcional na raiz do YAML mantendo `config/lab.yaml` intocado; validação estrita de `strategy` e `weights`; testes de contrato verdes | Nenhum (sincronizada com `tagging_method` de `upstream/develop` `923cd03`) | 2026-11-18 / 2026-11-26 · `55 passed, 1 skipped` + rebase `923cd03` limpo |
| **F3** | `[x] Concluída` | `100%` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Núcleo *Strategy* + `rrf` com `math.fsum` + telemetria `attributes.hybrid` + validação *NaN-free*; teste legado em `tests/test_modules.py` intacto e verde | Nenhum (concluída e auditada) | 2026-11-18 · `STRATEGIES`, `_score_rrf`, `_resolve_weights`, `_extract_raw_scores` e telemetria `g7-hybrid/1` verificados |
| **F4** | `[x] Concluída` | `100%` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `weighted_rrf` implementada; equivalência com pesos unitários, inversão de vencedor por peso e peso zero sem truncagem testados | Nenhum (concluída e auditada) | 2026-11-18 · `_score_weighted_rrf` e validação em runtime contra `payload["retrieval"]` verificados |
| **F5** | `[x] Concluída` | `100%` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `convex` (Min-Max normalizado por $\sum w_j$) implementada; invariância a escala, sensibilidade a *outlier* e casos degenerados ($\max = \min$, 1 item) testados | Nenhum (concluída e auditada) | 2026-11-18 · `_score_convex` com denominador $W$ sobre todas as chaves de `retrieval` verificado |
| **F6** | `[x] Concluída` | `100%` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `dbsf` ($\mu \pm 3\sigma$ com *clip* $[0, 1]$ normalizado por $\sum w_j$) implementada; $\sigma = 0$, *clip* em $n=12$ e soma exata `fsum(contributions) == fused` testados | Nenhum (unificada com `_resolve_hybrid_config` e 39 testes em `test_hybrid.py`) | 2026-11-18 / 2026-11-26 · `_score_dbsf`, `_resolve_hybrid_config` e invariância de escala no desempate verificados |
| **F7** | `[x] Concluída` | `100%` | `tests/test_pipeline.py` | Testes de composição ponta-a-ponta no `Pipeline` para todas as estratégias, bypass do `hybrid`, preservação de `metadata["tags"]` e do envelope JSON | Nenhum (concluída; suíte total com `121 passed, 2 skipped`) | 2026-11-18 / 2026-11-26 · `121 passed, 2 skipped` após rebase sobre `upstream/develop` `923cd03` |
| **F8** | `[~] Quase concluída` | `90%` | `experiments/g7/{corpus/,perguntas_gabarito.yaml,configs/*.yaml,README.md}` | Corpus autorizado (12 `.md`, 135 chunks), 24 perguntas (4 perfis), 35/35 trechos validados em chunk único, 13 YAMLs validados via `rag-lab validate` e piloto ao vivo (`L02` e `S02`) validado no Docker/Ollama | Pendente revisão humana cruzada de Diego e Gabriel Alves em `perguntas_gabarito.yaml` (`status_revisao: validado`) e congelamento do SHA-256 final | 2026-11-18 / 2026-11-26 · 12 docs + 24 perguntas + 13 configs + 10 envelopes de piloto ao vivo validados |
| **F9** | `[x] Concluída` | `100%` | `experiments/g7/{rodar.py,avaliar.py}`, `tests/test_g7_avaliacao.py` | Coletor online + avaliador com *replay* offline exato, tabelas T1–T6, planilha de avaliação humana e testes unitários das métricas; equivalência replay == online verificada (`10/10` no piloto) | Nenhum (pronta para Commit B) | 2026-11-26 · `133 passed, 2 skipped` + `--dry-run` e avaliação T1–T6 do piloto validados |
| **F10** | `[ ] Pendente` | `0%` | `compose.override.yaml` (local, não versionado), `data/results/g7/` | Banco dedicado `rag_g7` via Docker (`127.0.0.1:5432`), ingestão única com embeddings e execução oficial da matriz E0–E6 com `manifest.json` completos | Homologar revisão humana do gabarito (Commit C) e definir par oficial de modelos (`llama3.2:3b` + `embeddinggemma:300m` ou `lfm3:1.7b-q4_K_M` + `nomic-embed-text:latest`) | Aguardando homologação do gabarito (Commit C) |
| **F11** | `[ ] Pendente` | `0%` | `docs/g7-hybrid.md`, `experiments/g7/results/` | Avaliação humana (rubrica 0–2 de fidelidade/citações), relatório técnico final `docs/g7-hybrid.md` com tabelas T1–T6 e casos de *Fusion Tax*, suíte completa (`test_api.py`, `-m integration`, `npm run docs:check`) e PR | Depende dos resultados coletados na F10 | Aguardando F10 |

---

## 2. Arquitetura do `rag-lab` e Fronteira do G7

### 2.1 Fluxo de Dados Completo no `Pipeline.run`

O diagrama abaixo (compatível com [docs/diagrams/rag-pipeline.svg](diagrams/rag-pipeline.svg)) detalha o caminho exato do envelope JSON dentro de `src/rag_lab/pipeline.py`:

```text
+-----------------------------------------------------------------------------------+
| ENTRADA: Envelope(question, attributes={...})                                     |
| Pipeline.run() reinicia: queries=[question], retrieval={}, evidence=[],           |
|                          context="", answer=None, trace=[]                        |
| Remove apenas payload["attributes"].pop("hyde", None)                             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 1. PREPARAÇÃO DA CONSULTA (G1 -> G2 -> G3)                                        |
|   - multi_query  (G1): acrescenta até expanded_queries em payload["queries"]      |
|   - hyde         (G2): grava parágrafo sintético em payload["attributes"]["hyde"] |
|   - decomposicao (G3): acrescenta até expanded_queries em payload["queries"]      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. RECUPERAÇÃO INDEPENDENTE (G4, G5, G6) — via add_ranking(..., limit=candidate_k)|
|                                                                                   |
|   +-------------------+   +-------------------+   +---------------------------+   |
|   | tfidf (G4)        |   | bm25 (G5)         |   | embedding (G6)            |   |
|   | Cosseno em (0, 1] |   | BM25Okapi IDF>0   |   | pgvector 1 - (e <=> q)    |   |
|   | Filtra score > 0  |   | Escala (0, +inf)  |   | Escala [-1, 1], sem corte |   |
|   | Até candidate_k   |   | Filtra score > 0  |   | Inclui attributes["hyde"] |   |
|   +---------+---------+   +---------+---------+   +-------------+-------------+   |
|             |                       |                           |                 |
|             v                       v                           v                 |
|   retrieval["tfidf"]      retrieval["bm25"]           retrieval["embedding"]      |
|   (cada item traz item["score"] = s e item["scores"] = {nome_recuperador: s})     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. BIFURCAÇÃO ANTES DO SWITCH "hybrid" (pipeline.py, linhas 96-103)               |
|                                                                                   |
|   [SEMPRE EXECUTA, SEM CRONÔMETRO NO TRACE]                                       |
|   select_without_fusion(payload):                                                 |
|     - Intercala round-robin na ordem de inserção de payload["retrieval"]          |
|       (tfidf -> bm25 -> embedding)                                                |
|     - Deduplica por id, mescla item["scores"], mantém item["score"] do 1º método  |
|     - Grava resultado em payload["evidence"] (sem truncar para top_k)             |
|                                                                                   |
|   Switch modules.hybrid.enabled?                                                  |
|     |                                                                             |
|     +-- [false: status="bypass"] ---> Mantém payload["evidence"] do round-robin   |
|     |                                 (duration_ms ~ 0.0 ms no trace)             |
|     |                                                                             |
|     +-- [true:  status="executed"] -> Round-trip JSON + Hybrid().run()            |
|                                       + Envelope.model_validate()                 |
|                                       * SOBRESCREVE payload["evidence"]           |
|                                       * Grava item["score"] e scores["hybrid"]    |
|                                       * Grava payload["attributes"]["hybrid"]     |
|                                       * NÃO trunca para top_k                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. PÓS-RECUPERAÇÃO E PREPARAÇÃO DE CONTEXTO (G8 -> G9 -> G10)                     |
|   - reranking  (G8): calcula scores["reranking"] e REORDENA toda payload["evidence"]|
|                      por (-scores["reranking"], id)                               |
|   - sentence   (G9): preenche item["context_text"] via store.window(radius)       |
|   - compressao (G10): filtra sentenças de context_text/text com overlap lexical   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. CORTE FINAL E ORÇAMENTO: build_context(payload, settings)                      |
|   - Itera sobre payload["evidence"][:settings.pipeline.top_k] (padrão 5)          |
|   - Respeita max_context_chars (padrão 12000 caracteres)                          |
|   - Atribui item["citation"] = 1..n e substitui payload["evidence"] pelos         |
|     trechos efetivamente enviados ao LLM                                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 6. GERAÇÃO (LLM via Providers.chat)                                               |
|   - Se payload["context"] == "": retorna mensagem padrão sem chamar o LLM         |
|   - Caso contrário: gera payload["answer"] com citações [1], [2]...               |
+-----------------------------------------------------------------------------------+
```

### 2.2 Contratos Invioláveis do Repositório

1. **Assinatura e fronteira JSON (`src/rag_lab/modules/base.py`, `src/rag_lab/pipeline.py`):**
   - Todo módulo implementa `run(self, payload: dict, services: Services) -> dict`.
   - Em `pipeline.py` (linhas 102–103), cada módulo ativo recebe `json.loads(json.dumps(payload))` e seu retorno passa por `Envelope.model_validate(payload).model_dump(mode="json")`. Nenhum objeto não serializável em JSON, nem valores `NaN`/`Inf` (que quebram a serialização estrita do FastAPI em `api.py`), pode entrar no envelope.
2. **Validação estrita do YAML (`src/rag_lab/config.py`):**
   - Todos os modelos herdam de `StrictModel` (`model_config = ConfigDict(extra="forbid")` e `StrictBool` para switches). Qualquer chave não declarada no YAML aborta a carga com `ValidationError`.
3. **Preservação do envelope (`src/rag_lab/contracts.py`):**
   - `Hybrid.run` nunca deve sobrescrever `question`, `queries`, `retrieval`, `item["text"]` ou chaves alheias em `payload["attributes"]`.
   - `Hybrid.run` **não deve truncar** `payload["evidence"]` para `top_k`; o corte pertence exclusivamente a `build_context`.
4. **Trava de igualdade exata em `tests/test_modules.py` (linhas 10–24):**
   - O teste `test_rrf_rewards_agreement_without_adding_raw_scores` faz `assert` de igualdade exata no dicionário `result["evidence"][0]["scores"] == {"bm25": 1, "embedding": 0.9, "hybrid": pytest.approx(1 / 62 + 1 / 61)}`.
   - Portanto, `item["scores"]` **só pode conter** as chaves dos recuperadores que trouxeram o documento mais a chave `"hybrid"`. Qualquer decomposição de contribuições ou ranks por recuperador deve residir em `payload["attributes"]["hybrid"]`.
5. **Política de erro explícito vs. *bypass* (`docs/missoes.md`):**
   - O *bypass* é atribuição exclusiva do orquestrador quando `modules.hybrid.enabled` é `false`. Quando ativo, estados inválidos (como score bruto `NaN`/`Inf` ou `weights` incompatíveis com `payload["retrieval"]`) devem levantar `ValueError` explícito.

### 2.3 Acoplamentos Ocultos Identificados na Auditoria

| # | Acoplamento / Detalhe Oculto | Localização | Impacto para o G7 e Mitigação |
|---|---|---|---|
| **A1** | **`reranking` anula o efeito da fusão** | `src/rag_lab/modules/reranking.py` (linha 12) | Como nem `select_without_fusion` nem `Hybrid.run` truncam `payload["evidence"]`, ambos entregam exatamente o mesmo conjunto de candidatos (a união deduplicada de `retrieval`). O `Reranking` ordena toda a lista por `(-scores["reranking"], id)`, uma ordem total que descarta a ordenação anterior. Logo, com `reranking: true`, o `top_k` final é idêntico com `hybrid: true` ou `hybrid: false`. **Mitigação:** manter `reranking: false` nos blocos E1–E4 e demonstrar esse acoplamento empiricamente no bloco E5. |
| **A2** | **Assimetria de cronometragem entre Round-Robin e Hybrid** | `src/rag_lab/pipeline.py` (linhas 96–111) | `select_without_fusion(payload)` executa **sempre** na linha 97, **antes** de `start = perf_counter()` na linha 98. Quando `hybrid: false`, o `trace` registra `status: "bypass"` com `duration_ms ≈ 0.0`, sem contabilizar o tempo do *round-robin*. Quando `hybrid: true`, o `duration_ms` inclui o `json.loads(json.dumps(payload))`, a execução de `Hybrid.run` e o `Envelope.model_validate`. **Mitigação:** documentar essa assimetria na análise de latência (T4). |
| **A3** | **`item["score"]` no Round-Robin mistura escalas** | `src/rag_lab/pipeline.py` (linhas 46–49) | No *round-robin*, a primeira ocorrência faz `merged[item["id"]] = deepcopy(item)` e as ocorrências seguintes apenas atualizam `scores`. Assim, `evidence[i]["score"]` guarda o score bruto do primeiro recuperador que inseriu o trecho (TF-IDF em $(0, 1]$, BM25 em $(0, +\infty)$ ou Embedding em $[-1, 1]$). **Mitigação:** o avaliador offline nunca reordena a saída do *round-robin* por `score`; respeita a ordem posicional da lista. |
| **A4** | **Truncagem de `payload["evidence"]` em `build_context`** | `src/rag_lab/pipeline.py` (linha 73) | Ao final da consulta, `build_context` faz `payload["evidence"] = selected` (apenas os até `top_k` itens que couberam em `max_context_chars`). Os candidatos posicionados de `top_k + 1` em diante na lista fundida desapareceriam do envelope final. **Mitigação:** gravar a lista fundida completa (metadados compactos, sem duplicar `text`) em `payload["attributes"]["hybrid"]["ranking"]`. |
| **A5** | **Limpeza seletiva de `attributes` na entrada** | `src/rag_lab/pipeline.py` (linha 93) | `Pipeline.run` executa apenas `payload["attributes"].pop("hyde", None)`. Se `hybrid: false`, um `attributes.hybrid` enviado pelo cliente sobreviveria. **Mitigação:** em `Hybrid.run`, sobrescrever `payload["attributes"]["hybrid"]` integralmente; no avaliador, checar sempre o `trace` (`status == "executed"`). |
| **A6** | **Alvo de sintaxe `py311` no Ruff** | `pyproject.toml` (linha 38) | Embora o ambiente rode em Python 3.12 (`3.12.16`), o Ruff está configurado com `target-version = "py311"` e `select = ["E", "F", "I", "UP", "B"]`. **Mitigação:** não utilizar sintaxe exclusiva do Python 3.12 (PEP 695 `type Alias = ...`). |
| **A7** | **Bloqueio de *socketpair* do `TestClient` no sandbox Windows** | `tests/test_api.py` | `fastapi.testclient.TestClient` cria um *blocking portal* do `anyio` que, no Windows (`ProactorEventLoop`), invoca `socket.socketpair()` via loopback TCP, bloqueando no sandbox sem rede. **Mitigação:** rodar a suíte unitária no sandbox e executar `pytest` completo (incluindo `test_api.py` e integração) fora do sandbox na validação de entrega. |

---

## 3. Fundamentação Matemática e Decisões Arquiteturais (ADRs)

### 3.1 Limiar de Dominância do RRF com `candidate_k = 12` e `rrf_k = 60`

No RRF clássico (Cormack, Clarke & Büttcher, SIGIR 2009), o score de um documento $d$ é:

$$\text{RRF}(d) = \sum_{j \in R(d)} \frac{1}{k + r_j(d)}$$

onde $R(d)$ é o subconjunto de recuperadores que retornaram $d$ nas suas listas de tamanho até $C = \text{candidate\_k}$, e $r_j(d) \in \{1, \dots, C\}$ é a posição de $d$ no recuperador $j$.

Um documento presente na **última posição** ($r = C$) de $m$ listas supera estritamente qualquer documento presente na **primeira posição** ($r = 1$) de $m - 1$ listas sempre que:

$$\frac{m}{k + C} > \frac{m - 1}{k + 1} \iff m(k + 1) > (m - 1)(k + C) \iff k > m(C - 1) - C$$

Para o padrão do laboratório ($C = 12$):

| Comparação de consenso | Inequação ($C = 12$) | Limiar exato de empate | Situação no padrão `rrf_k = 60` |
|---|---|---|---|
| **$m = 2$ listas vs. $m = 1$ lista** | $k > 2(11) - 12 = 10$ | $k = 10$ ($2/22 = 1/11 \approx 0{,}09091$) | Dominância total ($2/72 \approx 0{,}02778 > 1/61 \approx 0{,}01639$) |
| **$m = 3$ listas vs. $m = 2$ listas** | $k > 3(11) - 12 = 21$ | $k = 21$ ($3/33 = 2/22 \approx 0{,}09091$) | Dominância total ($3/72 \approx 0{,}04167 > 2/61 \approx 0{,}03279$) |

**Consequências práticas para a missão G7:**
1. Com $k = 60$ e $C = 12$, a razão entre o 1º e o 12º lugar de uma mesma lista é de apenas $\frac{60+12}{60+1} = \frac{72}{61} \approx 1{,}18$. O RRF degenera em uma **votação por cardinalidade** ($|R(d)|$) seguida de desempate pela soma recíproca.
2. Esse efeito explica matematicamente o ***Fusion Tax***: um trecho irrelevante que aparece nas posições $(11, 12)$ de dois recuperadores vence um trecho perfeito que só um recuperador encontrou na posição $1$.
3. A varredura experimental de `rrf_k` precisa concentrar pontos na região de transição $k \in \{1, 3, 5, 10, 11, 15, 21, 22, 30, 60, 120, 1000\}$.

### 3.2 Viés de Ordem do Round-Robin e Viés 2:1 com Três Recuperadores

1. **Round-Robin (`select_without_fusion`):** seleciona na ordem $(r=1, \text{tfidf}) \to (r=1, \text{bm25}) \to (r=1, \text{embedding}) \to (r=2, \text{tfidf}) \dots$. Equivale a ordenar por $(\min_j r_j(d), \text{índice\_do\_recuperador})$, ignorando consenso (um documento em 1º nos três métodos não ganha prioridade sobre o 1º isolado do `tfidf`) e impondo um viés estático `tfidf > bm25 > embedding`. Em contrapartida, garante diversidade de representação no `top_k`, útil em perguntas compostas.
2. **Viés 2:1 (Lexical vs. Semântico):** `tfidf` e `bm25` compartilham exatamente o mesmo tokenizador (`tokens(text)` em `base.py`, linha 24: `re.findall(r"\w+", text.lower())`). Quando os três recuperadores estão ativos e $k > 10$, qualquer trecho com sobreposição lexical superficial capturado por `tfidf` e `bm25` ($m = 2$) supera **todos** os trechos puramente semânticos encontrados apenas por `embedding` ($m = 1$). Isso motiva diretamente a estratégia `weighted_rrf` (ex.: $w_{\text{tfidf}} = 1, w_{\text{bm25}} = 1, w_{\text{embedding}} = 2$).

### 3.3 Fusão Baseada em Score: `convex` (Min-Max) vs. `dbsf` ($\mu \pm 3\sigma$)

Para comparar fusão por posição (RRF) com fusão por magnitude de score (Bruch, Gai & Ingber, ACM TOIS 2023), implementamos duas estratégias normalizadas:

- **Combinação Convexa com Min-Max (`convex`):**
  - Re-escala os scores brutos de cada lista não vazia para $[0, 1]$ via $\hat{s}_j(d) = \frac{s_j(d) - s_{j,\min}}{s_{j,\max} - s_{j,\min}}$.
  - **Limitação estrutural:** em listas curtas ($n \le 12$), o 12º colocado recebe sempre $\hat{s}_j = 0$ (indistinguível de um documento fora da lista), e um único *outlier* de score no BM25 achata todos os candidatos do 2º ao 12º lugar para próximo de zero.
- **Distribution-Based Score Fusion (`dbsf`):**
  - Normaliza cada lista não vazia pela média amostral $\mu_j$ e desvio-padrão populacional $\sigma_j$ no intervalo $[\mu_j - 3\sigma_j, \mu_j + 3\sigma_j]$ com *clip* para $[0, 1]$:
    $$\hat{s}_j(d) = \min\left(1, \max\left(0, \frac{s_j(d) - (\mu_j - 3\sigma_j)}{6\sigma_j}\right)\right) = \text{clip}_{[0,1]}\left(0{,}5 + \frac{z_j(d)}{6}\right)$$
  - **Propriedade matemática (Desigualdade de Samuelson):** para $n$ elementos com desvio-padrão populacional $\sigma_j = \sqrt{\frac{1}{n}\sum (s_i - \mu)^2}$, o escore padronizado máximo satisfaz $|z_j| \le \sqrt{n - 1}$. Logo, para $n \le 10$ ($\sqrt{n-1} \le 3$), o *clip* nunca é ativado; para $n = 12$ ($\sqrt{11} \approx 3{,}317$), o *clip* atua apenas sobre um *outlier* extremo. Além disso, o último colocado da lista recebe $\hat{s}_j \approx 0{,}2\text{--}0{,}3 > 0$, preservando um **bônus de presença** em relação a documentos ausentes ($0$).

### 3.4 Precisão Numérica (`math.fsum`), Proteção *NaN-free* e Derivação de Recuperadores

1. **`math.fsum`:** a adição padrão em ponto flutuante IEEE 754 não é associativa ($(a + b) + c$ pode diferir de $(c + b) + a$ em $1\text{ ULP} \approx 10^{-17}$), o que quebraria empates simétricos entre três recuperadores por ruído numérico em vez do critério determinístico por `id`. Usamos `math.fsum` na soma das contribuições de cada documento.
2. **Denominador $\sum w_j$ e derivação pelas chaves de `payload["retrieval"]`:**
   - Em execuções via `Pipeline.run`, apenas os recuperadores ativos no YAML gravam chaves em `payload["retrieval"]` (mesmo quando retornam lista vazia `[]`, pois `add_ranking` sempre atribui `payload["retrieval"][name] = [...]`).
   - Em testes unitários diretos (`Hybrid().run(payload, services)`), o teste monta `payload["retrieval"]` diretamente sem necessariamente alterar os switches `services.settings.modules`.
   - Portanto, `Hybrid.run` deriva a lista de recuperadores participantes diretamente de `retrievers = list(payload["retrieval"].keys())`.
   - Se `services.settings.hybrid.weights` estiver preenchido (ou se `strategy == "weighted_rrf"`), `Hybrid.run` valida que `set(weights) == set(retrievers)` e que $\sum_{j \in \text{retrievers}} w_j > 0$, levantando `ValueError` explícito caso contrário.
   - Em `convex` e `dbsf`, a contribuição armazenada para cada recuperador $j$ já é pré-dividida por $W = \sum_{r \in \text{retrievers}} w_r$ (incluindo o peso de recuperadores em `payload["retrieval"]` cuja lista veio vazia `[]`), garantindo que `math.fsum(contributions.values()) == fused` feche com exatidão de ponto flutuante.

### 3.5 Tabela Consolidada de Decisões Aprovadas (D1 a D11 + Ajustes Técnicos)

| ID | Tema | Decisão Aprovada |
|---|---|---|
| **D1** | Localização no YAML | Seção raiz opcional `hybrid:` em `Settings` (`src/rag_lab/config.py`), mantendo `pipeline.rrf_k` no lugar original |
| **D2** | Rigor de `hybrid.weights` | Quando `modules.hybrid.enabled` é `true` no YAML: erro se `rrf` tiver `weights`; erro se `weighted_rrf` não tiver `weights`; chaves de `weights` devem coincidir com os recuperadores ativos e ter soma $> 0$. Em runtime (`Hybrid.run`), `weights` também é validado contra as chaves presentes em `payload["retrieval"]` |
| **D3** | Casos degenerados (`convex` / `dbsf`) | Quando $\max = \min$ (ou lista de 1 elemento) em `convex`, ou $\sigma = 0$ em `dbsf`, o score normalizado vale `1.0` (multiplicado por $w_j / \sum w_r$); documento ausente da lista recebe `0.0` |
| **D4** | *Theoretical Min-Max* | Mantido apenas como extensão opcional (*stretch*) após a conclusão da F6 |
| **D5** | Não-truncagem no `Hybrid` | `Hybrid.run` não trunca `payload["evidence"]`; o acoplamento com `reranking.py` é documentado e demonstrado no experimento E5 |
| **D6** | Intocabilidade de `pipeline.py` | Não editar `src/rag_lab/pipeline.py`; `Hybrid.run` sobrescreve `payload["attributes"]["hybrid"]` e o avaliador verifica `status == "executed"` no `trace` |
| **D7** | Isolamento de testes | Testes novos de fusão ficam em `tests/test_hybrid.py` para evitar conflitos de merge com os demais grupos em `tests/test_modules.py` |
| **D8** | Organização de experimentos | Centralizar scripts, configs, corpus (com licença/autoria), gabarito e resultados curados (CSV/relatório) em `experiments/g7/`; envelopes brutos ficam em `data/results/g7/` (ignorado pelo Git) |
| **D9** | Infraestrutura de banco | `uv` nativo no host Windows + PostgreSQL 16/pgvector via Docker com `compose.override.yaml` local (não versionado) expondo `127.0.0.1:5432` e banco dedicado `rag_g7` |
| **D10** | Escala da bateria experimental | 24 perguntas (6 por perfil); medir o tempo real do piloto na F9 para calibrar as 3 repetições nas configurações online |
| **D11** | Fluxo de branches | Trabalho na branch `feat/g7-hybrid` criada a partir de `develop`, com PR final para `develop` (respeitando `.github/workflows/main-source.yml`) |
| **AT1** | Fechamento exato de `contributions` | Em `convex` e `dbsf`, `ranking[].contributions[j]` armazena $c_j(d) = \frac{w_j}{\sum_{r \in \text{retrieval}} w_r} \cdot \hat{s}_j(d)$, garantindo `math.fsum(contributions.values()) == fused` |

---

## 4. Esquemas e Contratos do G7

### 4.1 Esquema YAML da Seção Opcional `hybrid:`

Quando a seção `hybrid:` é omitida (como no `config/lab.yaml` original), `Settings` assume `HybridConfig(strategy="rrf", weights={})`, mantendo 100% de retrocompatibilidade.

```yaml
# Exemplo 1: RRF clássico (padrão quando a seção hybrid é omitida)
pipeline:
  top_k: 5
  candidate_k: 12
  rrf_k: 60

hybrid:
  strategy: rrf

# Exemplo 2: Weighted RRF (corrige o viés 2:1 quando tfidf + bm25 + embedding estão ativos)
hybrid:
  strategy: weighted_rrf
  weights:
    tfidf: 1.0
    bm25: 1.0
    embedding: 2.0

# Exemplo 3: Combinação Convexa (Min-Max) com pesos iguais (weights omitido) ou explícitos
hybrid:
  strategy: convex
  weights:
    bm25: 0.35
    embedding: 0.65

# Exemplo 4: DBSF (Distribution-Based Score Fusion, mu +- 3*sigma)
hybrid:
  strategy: dbsf
  weights:
    bm25: 1.0
    embedding: 1.0
```

### 4.2 Tabela de Fórmulas e Casos Degenerados por Estratégia

Seja $R = \text{list(payload["retrieval"].keys())}$ o conjunto de recuperadores presentes no envelope, $w_j$ o peso efetivo do recuperador $j \in R$ ($w_j = 1{,}0$ quando `weights` está vazio, ou `weights[j]` quando informado) e $W = \sum_{j \in R} w_j$.

| Estratégia | Score normalizado $\hat{s}_j(d)$ para $d$ na posição $r_j \ge 1$ com score bruto $s_j(d)$ | Caso degenerado na lista $j$ | Contribuição $c_j(d)$ gravada em `contributions[j]` | Score final `fused` (`item["score"]` e `scores["hybrid"]`) |
|---|---|---|---|---|
| `rrf` | $\frac{1}{\text{rrf\_k} + r_j(d)}$ | Lista vazia $\Rightarrow$ sem itens | $\frac{1}{\text{rrf\_k} + r_j(d)}$ | $\text{fsum}(c_j(d))$ |
| `weighted_rrf` | $\frac{1}{\text{rrf\_k} + r_j(d)}$ | Peso $w_j = 0 \Rightarrow c_j(d) = 0{,}0$ (candidato permanece na lista com contribuição zero) | $\frac{w_j}{\text{rrf\_k} + r_j(d)}$ | $\text{fsum}(c_j(d))$ |
| `convex` | $\frac{s_j(d) - s_{j,\min}}{s_{j,\max} - s_{j,\min}} \in [0, 1]$ | Se $s_{j,\max} == s_{j,\min}$ (inclui lista com 1 item): $\hat{s}_j(d) = 1{,}0$ | $\frac{w_j}{W} \cdot \hat{s}_j(d)$ | $\text{fsum}(c_j(d)) \in [0, 1]$ |
| `dbsf` | $\min\left(1{,}0, \max\left(0{,}0, \frac{s_j(d) - (\mu_j - 3\sigma_j)}{6\sigma_j}\right)\right) \in [0, 1]$ | Se $\sigma_j == 0$ (inclui lista com 1 item): $\hat{s}_j(d) = 1{,}0$ | $\frac{w_j}{W} \cdot \hat{s}_j(d)$ | $\text{fsum}(c_j(d)) \in [0, 1]$ |

> **Nota:** Em todas as estratégias, se um documento $d$ não aparece na lista do recuperador $j$, sua contribuição naquele recuperador é $0$ e a chave $j$ não é incluída nos dicionários `ranks` e `contributions` daquele documento.

### 4.3 Esquema JSON da Telemetria em `payload["attributes"]["hybrid"]`

```json
{
  "schema": "g7-hybrid/1",
  "strategy": "convex",
  "params": {
    "rrf_k": 60,
    "weights": {
      "bm25": 1.0,
      "embedding": 1.0
    },
    "weight_sum": 2.0,
    "top_k": 5
  },
  "retrievers": {
    "bm25": {
      "candidates": 12,
      "weight": 1.0,
      "raw_min": 1.24,
      "raw_max": 9.81,
      "raw_mean": 3.42,
      "raw_std": 2.38
    },
    "embedding": {
      "candidates": 12,
      "weight": 1.0,
      "raw_min": 0.51,
      "raw_max": 0.78,
      "raw_mean": 0.62,
      "raw_std": 0.07
    }
  },
  "overlap": {
    "fused_count": 19,
    "in_all": 5,
    "exclusive": {
      "bm25": 7,
      "embedding": 7
    }
  },
  "boundary_tie": false,
  "ranking": [
    {
      "id": "8f14e45fceea167a5a36dedd4bea2543...",
      "source": "arquitetura.md",
      "position": 3,
      "fused": 0.9444444444444444,
      "fused_rank": 1,
      "ranks": {
        "bm25": 1,
        "embedding": 2
      },
      "contributions": {
        "bm25": 0.5,
        "embedding": 0.4444444444444444
      }
    }
  ]
}
```

- **Quando uma lista de recuperador está vazia (`candidates == 0`):** `raw_min`, `raw_max`, `raw_mean` e `raw_std` recebem `null` (`None` em Python), evitando qualquer geração de `NaN`.
- **`overlap.in_all`:** quantidade de documentos que apareceram em **todos** os recuperadores presentes em `payload["retrieval"]` (0 se `retrieval` estiver vazio ou se algum recuperador tiver lista vazia).
- **`overlap.exclusive[j]`:** quantidade de documentos que apareceram **apenas** no recuperador $j$.
- **`boundary_tie`:** `true` se e somente se `len(ranking) > top_k` e `math.isclose(ranking[top_k - 1]["fused"], ranking[top_k]["fused"], rel_tol=1e-12, abs_tol=1e-12)`, sinalizando que o corte de `build_context` na fronteira `top_k` foi decidido por empate de hash (`id`).

---

## 5. Registro Detalhado do que Já Foi Feito (F0 a F8)

### 5.1 Fase F0 — Preparação do Ambiente, Branch e Baseline de Qualidade

- **O que foi feito:**
  1. Instalação do gerenciador `uv` (`v0.13.9`) em `C:\Users\itofr\.local\bin\uv.exe`.
  2. Criação da branch local `develop` rastreando `origin/develop` e criação da branch de trabalho `feat/g7-hybrid` a partir do commit `e230073`.
  3. Sincronização do ambiente virtual `.venv` com CPython `3.12.16` e os 66 pacotes travados em `uv.lock` via `uv sync --frozen --extra dev --python 3.12`.
  4. Execução da bateria de validação sobre o repositório sem nenhuma alteração de código.
- **Arquivos criados/editados no repositório:** nenhum (apenas `.venv/`, ignorado pelo Git).
- **Por que foi feito assim:**
  - Garantir reprodutibilidade estrita com `--frozen` (sem alterar `uv.lock`) e isolar o trabalho do grupo na branch `feat/g7-hybrid` ramificada de `develop`, em conformidade com a workflow `.github/workflows/main-source.yml`.
- **Evidência de verificação:**
  - `rag-lab validate` -> `{"valid": true, ...}` (código de saída 0).
  - `rag-lab doctor --offline` -> `{"ok": true, "checks": [{"check": "python", "ok": true, "detail": "3.12.16"}, {"check": "dependencies", "ok": true, "detail": "Dependências instaladas e versões compatíveis"}]}`.
  - `ruff check .` -> `All checks passed!`.
  - `pytest` (módulos unitários `test_config.py`, `test_modules.py`, `test_pipeline.py`, `test_providers.py`, `test_ingestion.py`) -> `31 passed in 2.20s`.

### 5.2 Fase F1 — Testes de Caracterização em `tests/test_hybrid.py`

- **O que foi feito:**
  - Criação do arquivo `tests/test_hybrid.py` com 6 funções de teste (totalizando 9 casos parametrizados) que caracterizam matematicamente o comportamento do `Hybrid` original (`src/rag_lab/modules/hybrid.py`) e do `select_without_fusion` (`src/rag_lab/pipeline.py`).
- **Arquivos criados/editados:**
  - `tests/test_hybrid.py` (novo, 183 linhas).
- **Base do código implementado (`tests/test_hybrid.py`):**

```python
def _make_evidence(chunk: Chunk, retriever: str, score: float) -> dict:
    return Evidence(**chunk.model_dump(), score=score, scores={retriever: score}).model_dump()


@pytest.mark.parametrize(
    ("rrf_k", "expected_first"),
    [
        (1, "a"),   # 1/(1+1) = 0.5 > 2/(1+4) = 0.4 -> 1º isolado vence consenso mediano
        (60, "b"),  # 1/(60+1) < 2/(60+4) -> consenso vence 1º isolado
    ],
)
def test_rrf_k_parameter_shifts_consensus_vs_single_top_winner(services, rrf_k, expected_first):
    ...


@pytest.mark.parametrize(
    ("rrf_k", "expected_winner"),
    [
        (9, "a_top1"),     # 1/10 = 0.1000 > 2/21 = 0.0952
        (10, "a_top1"),    # 1/11 == 2/22 -> empate exato, desempate por id ("a_top1" < "b_rank12")
        (11, "b_rank12"),  # 2/23 = 0.0870 > 1/12 = 0.0833 (dominância m=2 sobre m=1)
    ],
)
def test_rrf_dominance_threshold_at_candidate_k_12(services, rrf_k, expected_winner):
    ...
```

- **Mapa completo dos testes de caracterização em `tests/test_hybrid.py`:**
  1. `test_rrf_classic_reference_behavior_and_exact_scores_keys`: verifica ordenação, valor de `score`, igualdade exata das chaves de `item["scores"]` (`{"bm25", "embedding", "hybrid"}`) e preservação de `attributes`.
  2. `test_rrf_k_parameter_shifts_consensus_vs_single_top_winner[1-a]` e `[60-b]`: comprova a inversão de vencedor entre $k=1$ e $k=60$.
  3. `test_rrf_dominance_threshold_at_candidate_k_12[9-a_top1]`, `[10-a_top1]` e `[11-b_rank12]`: comprova o limiar exato $k = 10$ para $C = 12$.
  4. `test_hybrid_does_not_truncate_to_top_k_and_does_not_mutate_retrieval`: comprova que `len(evidence) == 6` mesmo com `top_k = 2` e que `payload["retrieval"]` permanece idêntico a um `deepcopy` prévio.
  5. `test_hybrid_handles_empty_retriever_lists_and_ties_by_id`: verifica listas vazias e desempate lexicográfico por `id`.
  6. `test_select_without_fusion_round_robin_order_bias_and_score_preservation`: comprova que o *round-robin* prioriza a ordem de inserção (`tfidf -> bm25 -> embedding`) e mantém o `score` escalar do primeiro recuperador.
- **Por que foi feito assim:**
  - Isolar `tests/test_hybrid.py` (Decisão D7) evita conflito de merge em `tests/test_modules.py` com os outros 9 grupos e cria uma rede de segurança que garante que a refatoração da F3 preserve 100% do comportamento de referência quando `strategy == "rrf"`.
- **Evidência de verificação:**
  - `ruff check tests/test_hybrid.py` -> `All checks passed!`.
  - `pytest tests/test_hybrid.py -v` -> `9 passed in 1.42s` contra `src/rag_lab/modules/hybrid.py` original.

### 5.3 Fase F2 — Extensão Retrocompatível de `src/rag_lab/config.py` e `tests/test_config.py`

- **O que foi feito:**
  - Adição dos tipos `RETRIEVER_NAMES`, `RetrieverName`, `FusionStrategy`, `Weight` e do modelo `HybridConfig(StrictModel)` em `src/rag_lab/config.py`.
  - Inclusão do campo opcional `hybrid: HybridConfig = Field(default_factory=HybridConfig)` em `Settings` e das regras de validação cruzada em `Settings.validate_pipeline`.
  - Adição de 2 funções de teste (`test_hybrid_config_defaults_and_valid_strategies` e `test_invalid_hybrid_config_rejected` com 14 cenários parametrizados) em `tests/test_config.py`.
- **Arquivos criados/editados:**
  - `src/rag_lab/config.py` (+40/-5 linhas).
  - `tests/test_config.py` (+55 linhas).
  - `config/lab.yaml` mantido **intocado**.
- **Base do código implementado (`src/rag_lab/config.py`, linhas 35–42 e 79–115):**

```python
RETRIEVER_NAMES = ("tfidf", "bm25", "embedding")
RetrieverName = Literal["tfidf", "bm25", "embedding"]
FusionStrategy = Literal["rrf", "weighted_rrf", "convex", "dbsf"]
Weight = Annotated[
    StrictFloat | StrictInt,
    Field(ge=0, le=100, allow_inf_nan=False),
    AfterValidator(float),
]


class HybridConfig(StrictModel):
    strategy: FusionStrategy = "rrf"
    weights: dict[RetrieverName, Weight] = Field(default_factory=dict)


class Settings(StrictModel):
    database_url: str = Field(min_length=1, repr=False)
    llm: LLMConfig
    embeddings: EmbeddingConfig
    pipeline: PipelineConfig
    modules: dict[str, Switch]
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @model_validator(mode="after")
    def validate_pipeline(self) -> Settings:
        if set(self.modules) != set(MODULE_ORDER):
            raise ValueError(f"modules deve declarar exatamente: {', '.join(MODULE_ORDER)}")
        active_retrievers = {name for name in RETRIEVER_NAMES if self.modules[name].enabled}
        if self.modules["hybrid"].enabled:
            if len(active_retrievers) < 2:
                raise ValueError("Hybrid exige ao menos dois recuperadores ativos.")
            if self.hybrid.strategy == "rrf" and self.hybrid.weights:
                raise ValueError("A estratégia 'rrf' não aceita weights; use 'weighted_rrf'.")
            if self.hybrid.strategy == "weighted_rrf" and not self.hybrid.weights:
                raise ValueError("A estratégia 'weighted_rrf' exige weights explícitos.")
            if self.hybrid.weights:
                if set(self.hybrid.weights) != active_retrievers:
                    raise ValueError(
                        "hybrid.weights deve declarar exatamente os recuperadores ativos."
                    )
                if sum(self.hybrid.weights.values()) <= 0:
                    raise ValueError("hybrid.weights exige ao menos um peso maior que zero.")
        if self.modules["hyde"].enabled and not self.modules["embedding"].enabled:
            raise ValueError("HyDE exige Embedding ativo.")
        if self.pipeline.candidate_k < self.pipeline.top_k:
            raise ValueError("candidate_k deve ser maior ou igual a top_k.")
        return self
```

- **Por que foi feito assim:**
  1. **`StrictFloat | StrictInt` + `AfterValidator(float)`:** em Python, `bool` é subclasse de `int` (`isinstance(True, int) == True`), e um campo `float` padrão do Pydantic aceitaria `True` (convertendo para `1.0`) e strings como `"1.0"`. A união estrita `StrictFloat | StrictInt` rejeita `bool` e `str` na entrada, permite que o usuário escreva `1` ou `1.5` no YAML, e o `AfterValidator(float)` padroniza o dicionário resultante como `dict[str, float]`.
  2. **`allow_inf_nan=False` e `ge=0, le=100`:** impede que `.nan` ou `.inf` no YAML entrem na configuração e contaminem os cálculos de fusão.
  3. **Validação cruzada condicionada a `self.modules["hybrid"].enabled`:** evita quebrar testes ou execuções onde o usuário desliga `hybrid.enabled: false` (ex.: via helper `with_modules` de `tests/conftest.py`) mantendo uma seção `hybrid:` no YAML.
- **Evidência de verificação:**
  - `ruff check .` -> `All checks passed!`.
  - `rag-lab --config config/lab.yaml validate` -> `{"valid": true, ...}`.
  - `pytest --ignore=tests/test_api.py -v` -> `55 passed, 1 skipped in 2.05s`.

### 5.4 Fases F3 a F6 — Implementação Completa de `src/rag_lab/modules/hybrid.py` e Suíte `tests/test_hybrid.py`

- **O que foi feito e visão geral do diff:**
  - Reescrita modular de `src/rag_lab/modules/hybrid.py` (222 linhas, `+223/-12` no `git diff --stat`), implementando o padrão *Strategy* (`STRATEGIES: dict[str, Scorer]`) com 4 estratégias (`rrf`, `weighted_rrf`, `convex`, `dbsf`), soma de alta precisão com `math.fsum`, validação estrita de scores brutos e de pesos em runtime, e emissão da telemetria `g7-hybrid/1` em `payload["attributes"]["hybrid"]`.
  - Expansão de `tests/test_hybrid.py` (494 linhas totais, 15 funções de teste que se desdobram em 32 casos parametrizados).
- **Arquitetura real implementada em `src/rag_lab/modules/hybrid.py`:**
  1. **Tipo `Scorer`, funções de pontuação e registro `STRATEGIES` (linhas 13–62):**
     ```python
     Scorer = Callable[[list[float], float, float, int], list[float]]


     def _score_rrf(
         raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
     ) -> list[float]:
         return [1.0 / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]


     def _score_weighted_rrf(
         raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
     ) -> list[float]:
         return [weight / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]


     def _score_convex(
         raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
     ) -> list[float]:
         if not raw_scores or weight_sum <= 0.0:
             return []
         factor = weight / weight_sum
         lo = min(raw_scores)
         hi = max(raw_scores)
         span = hi - lo
         if span <= 0.0:
             return [factor for _ in raw_scores]
         return [factor * ((score - lo) / span) for score in raw_scores]


     def _score_dbsf(
         raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
     ) -> list[float]:
         if not raw_scores or weight_sum <= 0.0:
             return []
         factor = weight / weight_sum
         mean = fmean(raw_scores)
         std = pstdev(raw_scores)
         if std <= 0.0:
             return [factor for _ in raw_scores]
         lo = mean - 3.0 * std
         span = 6.0 * std
         return [factor * min(1.0, max(0.0, (score - lo) / span)) for score in raw_scores]


     STRATEGIES: dict[str, Scorer] = {
         "rrf": _score_rrf,
         "weighted_rrf": _score_weighted_rrf,
         "convex": _score_convex,
         "dbsf": _score_dbsf,
     }
     ```
  2. **Resolução e validação de pesos e extração *NaN-free* de scores (linhas 65–121):**
     ```python
     def _resolve_weights(
         strategy: str, configured_weights: dict[str, float], retrievers: list[str]
     ) -> dict[str, float]:
         if strategy not in STRATEGIES:
             raise ValueError(f"Estratégia híbrida desconhecida: {strategy}")
         if strategy == "rrf" and configured_weights:
             raise ValueError("A estratégia 'rrf' não aceita weights; use 'weighted_rrf'.")
         if strategy == "weighted_rrf" and not configured_weights:
             raise ValueError("A estratégia 'weighted_rrf' exige weights explícitos.")
         if not configured_weights:
             return {r: 1.0 for r in retrievers}
         if set(configured_weights) != set(retrievers):
             raise ValueError(
                 "hybrid.weights deve declarar exatamente os recuperadores presentes em retrieval."
             )
         weights: dict[str, float] = {}
         for r in retrievers:
             w = configured_weights[r]
             if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
                 raise ValueError(f"Peso inválido para o recuperador '{r}'.")
             weights[r] = float(w)
         if retrievers and math.fsum(weights.values()) <= 0.0:
             raise ValueError("hybrid.weights exige ao menos um peso maior que zero.")
         return weights


     def _extract_raw_scores(retriever: str, ranking: list[dict[str, Any]]) -> list[float]:
         raw_scores: list[float] = []
         for item in ranking:
             item_scores = item.get("scores") or {}
             raw = item_scores[retriever] if retriever in item_scores else item.get("score")
             if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                 raise ValueError(
                     f"Score bruto inválido em '{retriever}' para o item '{item.get('id')}'."
                 )
             raw_scores.append(float(raw))
         return raw_scores
     ```
  3. **Fluxo principal de `Hybrid.run` e montagem da telemetria `g7-hybrid/1` (linhas 124–222):**
     ```python
     class Hybrid:
         def run(self, payload: dict, services: Services) -> dict:
             settings = services.settings
             strategy = settings.hybrid.strategy
             rrf_k = settings.pipeline.rrf_k
             top_k = settings.pipeline.top_k

             retrieval = payload.get("retrieval") or {}
             retrievers = list(retrieval.keys())
             weights = _resolve_weights(strategy, settings.hybrid.weights, retrievers)
             weight_sum = math.fsum(weights.values())
             scorer = STRATEGIES[strategy]

             merged: dict[str, dict[str, Any]] = {}
             ranks_by_id: dict[str, dict[str, int]] = {}
             contribs_by_id: dict[str, dict[str, float]] = {}
             retriever_telemetry: dict[str, dict[str, Any]] = {}

             for retriever in retrievers:
                 ranking = retrieval[retriever]
                 weight = weights[retriever]
                 raw_scores = _extract_raw_scores(retriever, ranking)
                 retriever_telemetry[retriever] = _retriever_stats(raw_scores, weight)
                 contributions = scorer(raw_scores, weight, weight_sum, rrf_k)

                 for rank, (item, contribution) in enumerate(
                     zip(ranking, contributions, strict=True), start=1
                 ):
                     doc_id = item["id"]
                     if doc_id not in merged:
                         merged[doc_id] = deepcopy(item)
                         merged[doc_id]["score"] = 0.0
                         ranks_by_id[doc_id] = {}
                         contribs_by_id[doc_id] = {}
                     target = merged[doc_id]
                     target["scores"].update(item["scores"])
                     ranks_by_id[doc_id].setdefault(retriever, rank)
                     contribs_by_id[doc_id][retriever] = (
                         contribs_by_id[doc_id].get(retriever, 0.0) + contribution
                     )

             for doc_id, target in merged.items():
                 target["score"] = math.fsum(contribs_by_id[doc_id].values())

             ordered = sorted(merged.values(), key=lambda e: (-e["score"], e["id"]))
             for item in ordered:
                 item["scores"]["hybrid"] = item["score"]
             payload["evidence"] = ordered
     ```

- **Tratamento explícito de casos de borda implementados em `src/rag_lab/modules/hybrid.py`:**
  - **Score bruto `NaN`, `Inf`, `-Inf`, `bool`, `str` ou `None`:** `_extract_raw_scores` rejeita imediatamente com `ValueError("Score bruto inválido em ...")`, impedindo que valores não finitos derrubem o serializador JSON do FastAPI ou violem o contrato do `Envelope`.
  - **`weights` divergente, incompleto ou com soma zero:** `_resolve_weights` exige `set(configured_weights) == set(retrievers)` e `math.fsum(weights.values()) > 0.0`, levantando `ValueError` sem *fallback* silencioso.
  - **Recuperador com lista vazia `[]` em `payload["retrieval"]`:** `_retriever_stats` registra `candidates: 0` e `raw_min/raw_max/raw_mean/raw_std: None`, mas o peso desse recuperador permanece em `weight_sum = math.fsum(weights.values())`. Em `convex` e `dbsf`, isso preserva a penalização por ausência nos demais documentos.
  - **`payload["retrieval"]` completamente vazio (`{}`):**
    - Quando `strategy` é `"rrf"`, `"convex"` ou `"dbsf"` com `weights == {}` (padrão), `Hybrid.run` trata o caso **graciosamente**: `retrievers = []`, `weights = {}`, `weight_sum = 0.0`, `payload["evidence"] = []` e `payload["attributes"]["hybrid"]` é preenchido com `retrievers: {}`, `overlap: {"fused_count": 0, "in_all": 0, "exclusive": {}}`, `boundary_tie: False` e `ranking: []`.
    - Quando `strategy == "weighted_rrf"` (que exige `weights` não vazio) ou quando `weights` foi configurado com chaves que não existem em `retrieval == {}`, `_resolve_weights` levanta `ValueError` explícito.
  - **Listas constantes ou unitárias (`span <= 0.0` no `convex`, `std <= 0.0` no `dbsf`):** retornam `[factor for _ in raw_scores]` onde `factor = weight / weight_sum`, atribuindo score normalizado `1.0` sem divisão por zero.
  - **Imutabilidade, não-truncagem e pureza de `item["scores"]`:** cada documento único é clonado via `deepcopy(item)`, `payload["retrieval"]` jamais é mutado, `payload["evidence"]` mantém todos os candidatos únicos sem truncar para `top_k`, e `item["scores"]` recebe exclusivamente os scores dos recuperadores de origem mais `"hybrid"`.

- **Justificativa técnica das decisões de código:**
  1. **Derivação dos recuperadores via `retrievers = list(retrieval.keys())`:** desacopla `Hybrid.run` dos switches de `services.settings.modules` durante testes unitários diretos (como `tests/test_modules.py::test_rrf_rewards_agreement_without_adding_raw_scores`, onde o fixture padrão tem apenas `embedding: true`), enquanto em `Pipeline.run` `add_ranking` popula `payload["retrieval"][name]` para todos os recuperadores ativos (mesmo quando retornam `[]`).
  2. **Uso de `math.fsum` e o contraexemplo IEEE 754 `(1, 7, 2)` vs. `(2, 7, 1)` com `rrf_k = 60`:** em ponto flutuante binário de 64 bits, `((1/61 + 1/67) + 1/62) < ((1/62 + 1/67) + 1/61)` por 1 ULP. Com soma comum (`+=`), dois documentos com o mesmo multiconjunto de posições `{1, 2, 7}` em ordem diferente teriam o empate quebrado por erro de arredondamento (fazendo `"b"` vencer `"a"`) em vez da regra canônica por `id` (`"a" < "b"`). Com `math.fsum`, ambos produzem exatamente o mesmo `float`.
  3. **Pré-divisão das contribuições por `weight_sum` em `_score_convex` e `_score_dbsf`:** garante que `target["score"] = math.fsum(contribs_by_id[doc_id].values())` seja calculado a partir do mesmo dicionário exportado em `ranking[].contributions`, tornando o invariante `math.fsum(entry["contributions"].values()) == entry["fused"]` exato em todas as estratégias.
  4. **Dependências exclusivamente da stdlib (`math`, `statistics`, `copy`, `collections.abc`, `typing`):** mantém `uv.lock` 100% inalterado e respeita `target-version = "py311"` do Ruff.

- **Mapa completo dos 32 casos de teste em `tests/test_hybrid.py`:**

| Categoria | Função de Teste em `tests/test_hybrid.py` | Casos | O que valida |
|---|---|---|---|
| **RRF Clássico & Contratos (F1)** | `test_rrf_classic_reference_behavior_and_exact_scores_keys` | 1 | Ordem RRF (`rrf_k=60`), chaves exatas em `item["scores"]` e preservação de `attributes` |
| **Sensibilidade a `rrf_k` (F1)** | `test_rrf_k_parameter_shifts_consensus_vs_single_top_winner` | 2 | Inversão de vencedor entre `rrf_k=1` (1º isolado vence) e `rrf_k=60` (consenso 4º+4º vence) |
| **Limiar Matemático $C=12$ (F1)** | `test_rrf_dominance_threshold_at_candidate_k_12` | 3 | Transição em `rrf_k` $\in \{9, 10, 11\}$ para 1º isolado vs. 12º em duas listas |
| **Imutabilidade e Não-Truncagem (F1)** | `test_hybrid_does_not_truncate_to_top_k_and_does_not_mutate_retrieval` | 1 | `len(evidence) == 6` mesmo com `top_k=2`, `retrieval` intacto, preservação de `text` e `question` |
| **Listas Vazias e Empates (F1)** | `test_hybrid_handles_empty_retriever_lists_and_ties_by_id` | 1 | `evidence == []` quando tudo vazio e desempate lexicográfico `"a" < "b"` |
| **Round-Robin Baseline (F1)** | `test_select_without_fusion_round_robin_order_bias_and_score_preservation` | 1 | Viés `tfidf -> bm25 -> embedding` e manutenção do `score` bruto do 1º recuperador |
| **Telemetria `g7-hybrid/1` (F3)** | `test_telemetry_schema_boundary_tie_and_overwrites_injected_attribute` | 1 | Schema completo, `raw_* = None` em lista vazia, `boundary_tie=True` e sobrescrita de `attributes.hybrid` injetado |
| **Precisão IEEE 754 `fsum` (F3)** | `test_fsum_prevents_floating_point_order_bias_on_three_way_ties` | 1 | Igualdade exata na permutação `(1, 7, 2)` vs. `(2, 7, 1)` com `rrf_k=60` e desempate por `id` |
| **Proteção *NaN-free* (F3)** | `test_hybrid_rejects_non_finite_or_invalid_raw_scores` | 6 | `ValueError` para `nan`, `inf`, `-inf`, `True`, `"1.0"` e `None` |
| **Weighted RRF (F4)** | `test_weighted_rrf_unit_weights_match_classic_rrf_and_weights_shift_winner` | 1 | Equivalência com pesos `1.0`, correção do viés 2:1 com `w_embedding=2.0` e peso `0.0` sem truncar |
| **Validação de Pesos em Runtime (F4)** | `test_hybrid_runtime_weight_validation_errors` | 5 | `ValueError` para `rrf` com pesos, `weighted_rrf` sem pesos, chave faltante, chave extra e pesos todos zero |
| **Convex Min-Max (F5)** | `test_convex_scale_invariance_and_outlier_sensitivity` | 1 | Invariância a `bm25 * 1000` e compressão por *outlier* (`1002.0` vs. `6.0, 2.0`) |
| **Convex Degenerado & Denominador (F5)** | `test_convex_degenerate_lists_and_empty_retriever_denominator` | 1 | Lista unitária, scores iguais (`max == min`), recuperador vazio incluído em $W=3.0$ e `fsum(contributions) == fused` |
| **DBSF $\mu \pm 3\sigma$ (F6)** | `test_dbsf_zero_std_outlier_clipping_and_presence_bonus` | 1 | $\sigma=0$, *clip* em $1.0$ para *outlier* em $n=12$ ($z = \sqrt{11} > 3$) e bônus de presença ($>0$) na cauda |
| **Contrato JSON & Envelope (F6)** | `test_all_strategies_roundtrip_strict_json_and_envelope_contract` | 6 | `json.dumps(..., allow_nan=False)` + `Envelope.model_validate` + `fsum(contributions) == fused` nas 4 estratégias |

### 5.5 Fase F7 — Testes de Composição Ponta-a-Ponta em `tests/test_pipeline.py`

- **O que foi feito:**
  - Adição de 2 funções de teste (5 casos parametrizados, +45 linhas) em `tests/test_pipeline.py` exercitando o `Pipeline.run` completo com `Hybrid` ativo nas 4 estratégias e em modo `bypass`.
- **Trecho implementado em `tests/test_pipeline.py` (linhas 92–134):**
  ```python
  @pytest.mark.parametrize(
      ("strategy", "weights"),
      [
          ("rrf", {}),
          ("weighted_rrf", {"bm25": 1.0, "embedding": 2.0}),
          ("convex", {}),
          ("dbsf", {"bm25": 1.0, "embedding": 1.0}),
      ],
  )
  def test_pipeline_hybrid_strategies_emit_telemetry_and_preserve_contract(
      services, strategy, weights
  ):
      services.settings = with_modules(services.settings, "bm25", "embedding", "hybrid")
      services.settings.hybrid.strategy = strategy
      services.settings.hybrid.weights = weights
      services.settings.pipeline.top_k = 2

      result = Pipeline(services).run(
          Envelope(question="RAG documentos", attributes={"grupo": "G7"})
      )
      trace_by_module = {entry["module"]: entry for entry in result.trace}

      assert trace_by_module["hybrid"]["status"] == "executed"
      assert result.attributes["grupo"] == "G7"
      telemetry = result.attributes["hybrid"]
      assert telemetry["schema"] == "g7-hybrid/1"
      assert telemetry["strategy"] == strategy
      assert len(telemetry["ranking"]) >= len(result.evidence)
      assert len(result.evidence) <= 2
      assert "hybrid" in result.evidence[0].scores
      assert result.model_dump_json()


  def test_pipeline_without_hybrid_uses_round_robin_and_leaves_hybrid_bypassed(services):
      services.settings = with_modules(services.settings, "tfidf", "bm25", "embedding")
      result = Pipeline(services).run(Envelope(question="RAG documentos"))
      trace_by_module = {entry["module"]: entry for entry in result.trace}

      assert trace_by_module["hybrid"]["status"] == "bypass"
      assert "hybrid" not in result.attributes
      assert result.evidence
      assert "hybrid" not in result.evidence[0].scores
  ```
- **Evidência consolidada de verificação ao final da F7:**
  - `git diff --stat`:
    ```text
     src/rag_lab/config.py         |  45 ++++++++-
     src/rag_lab/modules/hybrid.py | 223 +++++++++++++++++++++++++++++++++++++++---
     tests/test_config.py          |  55 +++++++++++
     tests/test_pipeline.py        |  45 +++++++++
     4 files changed, 351 insertions(+), 17 deletions(-)
    ```
  - `uv run ruff check .` -> `All checks passed!`
  - `uv run rag-lab --config config/lab.yaml validate` -> `{"valid": true, ...}`
  - `uv run pytest --ignore=tests/test_api.py -v` -> `83 passed, 1 skipped in 2.03s` (nota: `tests/test_api.py` permanece intacto e será executado fora do sandbox na validação de entrega devido ao `socket.socketpair()` do `TestClient`/`anyio` no Windows).

### 5.6 Fase F8 — Corpus Autoral, Gabarito Independente e Configurações YAML em `experiments/g7/`

#### 5.6.1 Árvore de Diretórios e Isolamento Arquitetural em `experiments/g7/`

Como `.gitignore` (linhas 259–261) ignora `data/documents/*` e `data/results/`, centralizamos todos os artefatos reprodutíveis do protocolo experimental em `experiments/g7/` (Decisão D8). Isso garante que o corpus autoral, o gabarito anotado, os 13 manifestos YAML e a documentação do protocolo sejam versionados no Git, enquanto os envelopes JSON brutos das execuções futuras ficarão isolados em `data/results/g7/`:

```text
experiments/g7/
├── README.md                              # Declaração CC-BY-4.0, mapa do corpus, matriz YAML e protocolo de revisão
├── perguntas_gabarito.yaml                # 24 perguntas (6 por perfil) e 35 trechos relevantes anotados manualmente
├── corpus/                                # 12 documentos .md UTF-8 (23.991 caracteres, 135 chunks sentenciais)
│   ├── 01_arquitetura_geral.md
│   ├── 02_autenticacao_identidade.md
│   ├── 03_banco_vetorial_indexacao.md
│   ├── 04_filas_mensageria.md
│   ├── 05_faturamento_cotas.md
│   ├── 06_seguranca_criptografia.md
│   ├── 07_backup_recuperacao.md
│   ├── 08_auditoria_conformidade.md
│   ├── 09_observabilidade_telemetria.md
│   ├── 10_slas_disponibilidade.md
│   ├── 11_deploy_configuracao.md
│   └── 12_governanca_ingestao_dados.md
└── configs/                               # 13 configurações YAML validadas via `rag-lab validate`
    ├── e0_tfidf.yaml
    ├── e0_bm25.yaml
    ├── e0_embedding.yaml
    ├── e1_rr_bm25_emb.yaml
    ├── e1_rr_todos.yaml
    ├── e2_rrf_k10_bm25_emb.yaml
    ├── e2_rrf_k60_bm25_emb.yaml
    ├── e2_rrf_k60_todos.yaml
    ├── e4_weighted_rrf_todos.yaml
    ├── e4_convex_bm25_emb.yaml
    ├── e4_dbsf_bm25_emb.yaml
    ├── e5_coupling_rr_rerank.yaml
    └── e5_coupling_rrf_rerank.yaml
```

#### 5.6.2 Engenharia do Corpus ("Plataforma Núcleo Aurora" — 12 Documentos, 23.991 Caracteres, 135 Chunks)

- **Por que construímos um domínio 100% fictício e autoral (licença CC-BY-4.0):**
  1. **Eliminação de vazamento de memória paramétrica do LLM (`llama3.2:3b`):** se o corpus descrevesse sistemas públicos reais (como documentação padrão de Kubernetes ou PostgreSQL), o modelo gerador poderia responder corretamente mesmo quando a recuperação falhasse. Ao criar a plataforma fictícia **"Núcleo Aurora"** com normas (`POL-SEG-09`, `POL-SEG-14`, `POL-AUD-03`, `POL-SLA-01`), códigos de erro (`ERR-AUR-417`, `ERR-AUR-508`, `ERR-AUR-429`, `SEC-LCK-301`, `ALR-LAT-901`) e parâmetros operacionais próprios, garantimos que a qualidade da geração dependa estritamente das evidências entregues no `top_k`.
  2. **Conformidade ética e autoral (`docs/missoes.md`):** o acervo foi inteiramente redigido pelo Grupo G7 (Diego, Gabriel Alves e João Victor Póvoa) e licenciado sob **CC-BY-4.0**, sem dados pessoais e com autorização explícita para uso e publicação no repositório.
  3. **Compatibilidade determinística com `split_document(..., max_chars=1000)` (`src/rag_lab/ingestion.py`):** todas as sentenças foram redigidas em linha única abaixo de `1000` caracteres e sem abreviações internas seguidas de espaço (como `ex. ` ou `Sr. `), garantindo que a expressão regular `re.split(r"(?<=[.!?])\s+|\n+", text)` produza exatamente `1` chunk por sentença ou título (`135` chunks no total).

- **Tabela completa dos 12 documentos do corpus (`experiments/g7/corpus/`):**

| Arquivo | Caracteres | Chunks (`split_document`, 1000 chars) | SHA-256 (prefixo 12 hex) | Papel Experimental e Mecanismos Plantados |
|---|---|---|---|---|
| `01_arquitetura_geral.md` | 2.234 | 13 | `4c803007a13b` | Portas (`9443`, `5353`), limite `380` conexões, coordenador `aurora-core`; fato complementar para `S04` e `C06` |
| `02_autenticacao_identidade.md` | 2.066 | 12 | `f7ddef0b5595` | `POL-SEG-09` (`18` min), `SEC-LCK-301` (`5` tentativas), `AUR-ID-702`, rotação Ed25519 (`14` dias); distrator de "token" e "rotação" (`L01`, `C01`) |
| `03_banco_vetorial_indexacao.md` | 2.039 | 11 | `b903f81353dd` | `ERR-AUR-417`, `768` dimensões, busca exata até `50000`, rollback atômico; par redundante com `07` e alvo *near-miss* `N02` |
| `04_filas_mensageria.md` | 1.826 | 11 | `8012327594e2` | `ERR-AUR-508`, `aurora.dlq.critico` (`4` tentativas), expurgo em `21` dias, janela `120` s; distrator lexical de "janela" e "expurgo" (`L03`) |
| `05_faturamento_cotas.md` | 1.844 | 11 | `f413ef7747cd` | Franquia `4500000` tokens, `ERR-AUR-429`, isenção sem geração, custo do preflight; distrator de "token" (`L04`, `S04`, `C05`, `N03`) |
| `06_seguranca_criptografia.md` | 2.096 | 11 | `7266eec7b719` | `POL-SEG-14` (`aurora_db_net`, porta `5432`), recifragem semestral `aurora-vault`, sanitização HTTP `503` (`L05`, `S02`, `C01`) |
| `07_backup_recuperacao.md` | 1.901 | 11 | `56616bc55ebb` | Retenção WORM `90` dias, RPO `15` min, RTO `40` min, protocolo `DR-SIM-04`, transação atômica por documento (`S01`, `C02`, `C03`) |
| `08_auditoria_conformidade.md` | 1.941 | 11 | `71f80bf3aeb8` | `POL-AUD-03`, expurgo em `365` dias, retenção fiscal `5` anos, proibição de gabarito automático (`S06`, `C02`) |
| `09_observabilidade_telemetria.md` | 1.961 | 11 | `e41d011c5593` | OpenTelemetry porta `4317`, rotação de log `500` MB, `ALR-LAT-901` (`4500` ms), rota `/health` sem LLM (`L06`, `C05`, `C06`) |
| `10_slas_disponibilidade.md` | 1.958 | 11 | `f549c8cbf295` | `POL-SLA-01`, `99,95%` (`21` min/mês), crédito de `12%`, janela terças `02h00–04h00` (`C03`, distrator de `N03`) |
| `11_deploy_configuracao.md` | 2.065 | 11 | `3ce0076a485b` | `AURORA_CONFIG_PATH`, ausência de hot-reload (exige reinício), booleanos estritos (`S03`) |
| `12_governanca_ingestao_dados.md` | 2.060 | 11 | `d29201085777` | `.txt`/`.md` UTF-8, identidade por caminho relativo, obrigatoriedade de reingestão (`S05`, `C04`, `N01`) |
| **Total** | **23.991** | **135** | — | — |

- **Mapeamento dos 5 mecanismos de estresse experimental plantados no corpus:**
  1. **Identificadores e códigos exatos (força lexical de `bm25`/`tfidf`):** códigos de falha (`ERR-AUR-417` em `03`, `ERR-AUR-508` em `04`, `ERR-AUR-429` em `05`, `SEC-LCK-301` em `02`, `ALR-LAT-901` em `09`), normas (`POL-SEG-09` em `02`, `POL-SEG-14` em `06`, `POL-AUD-03` em `08`, `POL-SLA-01` em `10`), variáveis/filas (`AURORA_CONFIG_PATH` em `11`, `aurora.dlq.critico` em `04`, `aurora_db_net` em `06`) e portas (`9443`, `5353`, `5432`, `4317`, `8000`).
  2. **Descrições formais para recuperação semântica por paráfrase (força de `embedding`):** sentenças redigidas com vocabulário técnico formal para que as perguntas `S01`–`S06` utilizem sinônimos cotidianos (ex.: perguntar sobre *"arquivo salvo pela metade"* em `S01` para recuperar *"transação inteira sofre reversão automática"* em `03` e *"nenhuma metade de arquivo fica gravada"* em `07`; perguntar sobre *"descubra endereços de servidores internos ou credenciais"* em `S02` para recuperar *"convertida em erro genérico 503 sem exibir URLs ou chaves de API"* em `06`).
  3. **Distratores lexicais cruzados entre domínios (indução controlada de *Fusion Tax*):**
     - Termo `"token"` / `"tokens"`: ocorre em `02_autenticacao_identidade.md` (token de sessão de 18 minutos) e em `05_faturamento_cotas.md` (franquia de 4.500.000 tokens de inferência e tarifa por lote de mil tokens).
     - Termo `"janela"` / `"janelas"`: ocorre em 5 documentos distintos — tolerância de 30 s na renovação de sessão (`02`), desduplicação de 120 s no NATS (`04`), tarifação móvel de 60 s (`05`), agregação de percentis `p95`/`p99` de 30 s (`09`) e manutenção programada às terças 02h00–04h00 (`10`).
     - Termo `"rotação"`: ocorre na rotação de chaves Ed25519 a cada 14 dias (`02`) e na rotação de arquivos de log locais a cada 500 MB (`09`).
     - Termos `"expurgo"` / `"retenção"`: ocorrem na retenção de 21 dias na DLQ antes do expurgo (`04`), na retenção WORM de 90 dias de backups (`07`), no expurgo de metadados transitórios em 365 dias (`08`) e na retenção fiscal de 5 anos (`08`).
  4. **Pares quase redundantes em arquivos distintos (exercício de consenso e Jaccard):**
     - Atomicidade transacional por documento e preservação da versão anterior em caso de falha de embeddings (`03_banco_vetorial_indexacao.md` e `07_backup_recuperacao.md`).
     - Limiar de `4500` milissegundos de tempo de resposta/geração (`09_observabilidade_telemetria.md` e `10_slas_disponibilidade.md`).
     - Proibição de segredos no YAML e referência por variáveis de ambiente (`06_seguranca_criptografia.md` e `11_deploy_configuracao.md`).
  5. **Fatos complementares divididos em documentos distintos (perguntas compostas `C01`–`C06`):** pares de informações onde nenhum documento isolado contém a resposta completa, exigindo que a estratégia de seleção/fusão posicione ambos os documentos dentro do corte `top_k: 5`.

#### 5.6.3 Design e Governança do Gabarito (`experiments/g7/perguntas_gabarito.yaml`)

- **Mapa completo das 24 perguntas por perfil (`6` literais, `6` semânticas, `6` compostas, `6` sem resposta):**

| ID | Perfil | `responder` | Documentos-alvo (`documentos_relevantes`) | Qtd. Trechos | Anotador / Revisor | O que estressa experimentalmente |
|---|---|---|---|---|---|---|
| `L01` | `literal` | `true` | `02_autenticacao_identidade.md` | 2 | Diego / Gabriel Alves | Códigos `POL-SEG-09` e `SEC-LCK-301` com distrator "token" em `05` |
| `L02` | `literal` | `true` | `03_banco_vetorial_indexacao.md` | 1 | Diego / Gabriel Alves | Código exato `ERR-AUR-417` frente a `ERR-AUR-508` e `ERR-AUR-429` |
| `L03` | `literal` | `true` | `04_filas_mensageria.md` | 2 | Gabriel Alves / João Victor | Fila `aurora.dlq.critico` (21 dias) com distrator "expurgo" em `07` e `08` |
| `L04` | `literal` | `true` | `05_faturamento_cotas.md` | 2 | Gabriel Alves / Diego | Franquia `4500000` tokens e `ERR-AUR-429` com distrator "bloqueio/token" em `02` |
| `L05` | `literal` | `true` | `06_seguranca_criptografia.md` | 1 | João Victor / Diego | `POL-SEG-14`, `aurora_db_net` e porta `5432` (*outlier* de score no BM25) |
| `L06` | `literal` | `true` | `09_observabilidade_telemetria.md` | 2 | João Victor / Gabriel Alves | `ALR-LAT-901` (`4500` ms) e porta `4317` com distrator `4500` ms em `10` |
| `S01` | `semantica` | `true` | `03_banco_vetorial_indexacao.md`, `07_backup_recuperacao.md` | 3 | Diego / Gabriel Alves | Paráfrase de rollback/transação atômica; consenso em par redundante `03`+`07` |
| `S02` | `semantica` | `true` | `06_seguranca_criptografia.md` | 1 | Diego / João Victor | Paráfrase de sanitização de exceções em HTTP 503; testa viés 2:1 contra `embedding` |
| `S03` | `semantica` | `true` | `11_deploy_configuracao.md` | 1 | Gabriel Alves / Diego | Paráfrase de ausência de hot-reload sem usar "recarga", "switches" ou "YAML" |
| `S04` | `semantica` | `true` | `05_faturamento_cotas.md`, `01_arquitetura_geral.md` | 2 | Gabriel Alves / João Victor | Paráfrase de isenção de tarifação quando não há evidências na base |
| `S05` | `semantica` | `true` | `12_governanca_ingestao_dados.md` | 2 | João Victor / Diego | Paráfrase de identidade por caminho relativo (renomear duplica; deletar mantém) |
| `S06` | `semantica` | `true` | `08_auditoria_conformidade.md` | 1 | João Victor / Gabriel Alves | Paráfrase da proibição de usar saída automática do sistema como gabarito |
| `C01` | `composta` | `true` | `02_autenticacao_identidade.md`, `06_seguranca_criptografia.md` | 2 | Diego / Gabriel Alves | Rotação Ed25519 (14 dias) vs. recifragem `aurora-vault` (semestral) + distrator em `09` |
| `C02` | `composta` | `true` | `07_backup_recuperacao.md`, `08_auditoria_conformidade.md` | 3 | Diego / João Victor | Retenção WORM (90 dias) vs. expurgo auditoria (365 dias) vs. fiscal (5 anos) |
| `C03` | `composta` | `true` | `07_backup_recuperacao.md`, `10_slas_disponibilidade.md` | 3 | Gabriel Alves / Diego | RPO (15 min) e RTO (40 min) vs. orçamento SLA de indisponibilidade (21 min/mês) |
| `C04` | `composta` | `true` | `03_banco_vetorial_indexacao.md`, `12_governanca_ingestao_dados.md` | 2 | Gabriel Alves / João Victor | Cálculo SHA-256 da chave vetorial (`03`) + regra de reingestão obrigatória (`12`) |
| `C05` | `composta` | `true` | `09_observabilidade_telemetria.md`, `05_faturamento_cotas.md` | 2 | João Victor / Diego | Rota `/health` sem LLM (`09`) vs. preflight consumindo tokens reais (`05`) |
| `C06` | `composta` | `true` | `01_arquitetura_geral.md`, `09_observabilidade_telemetria.md` | 3 | João Victor / Gabriel Alves | Portas Envoy (`9443`), DNS interno (`5353`) e OpenTelemetry (`4317`) |
| `N01` | `sem_resposta` | `false` | `[]` | 0 | Diego / Gabriel Alves | *Near-miss* 1: pede DPI/idioma de OCR no `aurora-ingest` (OCR é bloqueado em `12`) |
| `N02` | `sem_resposta` | `false` | `[]` | 0 | Gabriel Alves / João Victor | *Near-miss* 2: pede `m` e `ef_construction` HNSW até 50000 trechos (`03` usa busca exata) |
| `N03` | `sem_resposta` | `false` | `[]` | 0 | João Victor / Diego | *Near-miss* 3: pede desconto do "plano Estudantil Básico" (distrator crédito 12% SLA) |
| `N04` | `sem_resposta` | `false` | `[]` | 0 | Diego / João Victor | Fora do domínio 1 (esportes): escalação do Brasil na final da Copa de 2002 |
| `N05` | `sem_resposta` | `false` | `[]` | 0 | Gabriel Alves / Diego | Fora do domínio 2 (culinária): temperatura e fermentação de panetone artesanal |
| `N06` | `sem_resposta` | `false` | `[]` | 0 | João Victor / Gabriel Alves | Fora do domínio 3 (biologia): fixação noturna de carbono na fotossíntese CAM |

- **Trecho real de `experiments/g7/perguntas_gabarito.yaml` (exemplo respondível `S02` e exemplo sem resposta *near-miss* `N02`):**

```yaml
  - id: S02
    perfil: semantica
    pergunta: "De que forma o serviço web impede que um usuário externo descubra endereços de servidores internos ou credenciais ao ocorrer um defeito de comunicação?"
    responder: true
    documentos_relevantes:
      - 06_seguranca_criptografia.md
    trechos_relevantes:
      - fonte: 06_seguranca_criptografia.md
        trecho: "Para evitar vazamento de topologia e de segredos em ambientes de produção, qualquer exceção lançada por bibliotecas de clientes externos é interceptada na borda HTTP e convertida em erro genérico 503 sem exibir URLs ou chaves de API."
        essencial: true
    fatos_esperados:
      - "Exceções de bibliotecas/SDKs são interceptadas na borda HTTP"
      - "Conversão em erro genérico HTTP 503 omitindo URLs e chaves de API"
    observacao_experimental: "Consulta conceitual onde métodos lexicais têm baixa sobreposição e o recuperador vetorial (embedding) deve liderar; testa o viés 2:1 do RRF."
    anotador: Diego
    revisor: João Victor Póvoa
    status_revisao: pendente_validacao_humana

  - id: N02
    perfil: sem_resposta
    pergunta: "Quais valores de m e ef_construction são recomendados para calibrar o índice aproximado HNSW no aurora-vectordb em coleções de até 50000 fragmentos?"
    responder: false
    documentos_relevantes: []
    trechos_relevantes: []
    fatos_esperados:
      - "Abstenção explícita: a base não contém parâmetros HNSW (m ou ef_construction), pois até 50000 fragmentos o aurora-vectordb utiliza busca exata sem índice ANN."
    observacao_experimental: "Near-miss 2: alta sobreposição lexical com 03_banco_vetorial_indexacao.md ('aurora-vectordb', '50000 fragmentos', 'índice aproximado'), testando se o LLM inventa parâmetros HNSW."
    anotador: Gabriel Alves
    revisor: João Victor Póvoa
    status_revisao: pendente_validacao_humana
```

- **Por que usamos casamento por substring normalizada dentro de chunk único de `split_document(..., 1000)`:**
  - Em `src/rag_lab/ingestion.py` (linhas 24–25), o `id` de cada chunk é `sha256(f"{document_id}:{i}:{part}")`. Se uma sentença não relacionada no início do documento fosse editada, o índice `i` e o `id` de todas as sentenças seguintes mudariam.
  - Ao anotar `trechos_relevantes` pelo texto literal da sentença normalizada por espaço (`re.sub(r"\s+", " ", s.strip())`) + `fonte`, e validar que cada trecho está contido integralmente dentro de um único chunk gerado por `split_document(source, text, 1000)`, a avaliação de `Recall@k`, `MRR` e `nDCG@k` por trecho torna-se imune a mudanças de hash/posição e garante que nenhum trecho relevante foi partido ao meio na ingestão.
- **Protocolo de revisão humana cruzada e congelamento do hash SHA-256:**
  - Todos os 24 itens nascem com `status_revisao: pendente_validacao_humana` e pares distintos `(anotador, revisor)` distribuídos entre Diego, Gabriel Alves e João Victor Póvoa.
  - Antes da execução da matriz experimental na Fase F10, os três integrantes revisam os itens, alteram `status_revisao` para `"homologado"` (ou `"validado"`) e congelam o SHA-256 do arquivo `perguntas_gabarito.yaml`, que será gravado automaticamente em cada `manifest.json` pelo `rodar.py`.

#### 5.6.4 Matriz de Configurações Experimentais (`experiments/g7/configs/` — 13 YAMLs)

- **Variáveis mantidas estritamente constantes nos 13 arquivos:**
  - `database_url: ${DATABASE_URL:-postgresql://rag:rag@localhost:5432/rag_g7}` (sem segredos hardcoded);
  - `llm`: `provider: ollama`, `base_url: ${OLLAMA_BASE_URL:-http://localhost:11434}`, `model: llama3.2:3b`, `api_key_env: LLM_API_KEY`, `temperature: 0.0`, `timeout: 120`;
  - `embeddings`: `provider: ollama`, `base_url: ${OLLAMA_BASE_URL:-http://localhost:11434}`, `model: nomic-embed-text`, `api_key_env: EMBEDDING_API_KEY`, `dimensions: 768`, `timeout: 120`;
  - `pipeline`: `top_k: 5`, `candidate_k: 12`, `expanded_queries: 3`, `sentence_radius: 1`, `max_context_chars: 12000`, `max_sentence_chars: 1000`;
  - `modules`: `multi_query: false`, `hyde: false`, `decomposicao: false`, `sentence: false`, `compressao: false` em todos os 13 arquivos; `reranking: false` em E0–E4 e `reranking: true` exclusivamente nos 2 arquivos de E5.

- **Tabela completa dos 13 YAMLs experimentais:**

| Arquivo YAML | Bloco | Recuperadores Ativos | `hybrid` | Estratégia / Parâmetros (`rrf_k`, `weights`) | `reranking` | Hipótese / Objetivo Experimental |
|---|---|---|---|---|---|---|
| `e0_tfidf.yaml` | `E0` | `tfidf` | `false` | — (`rrf_k: 60`) | `false` | Medir baseline lexical puro por cosseno TF-IDF |
| `e0_bm25.yaml` | `E0` | `bm25` | `false` | — (`rrf_k: 60`) | `false` | Medir baseline lexical BM25Okapi (IDF Lucene positivo) |
| `e0_embedding.yaml` | `E0` | `embedding` | `false` | — (`rrf_k: 60`) | `false` | Medir baseline semântico puro (`pgvector` cosseno exato) |
| `e1_rr_bm25_emb.yaml` | `E1` | `bm25`, `embedding` | `false` | `select_without_fusion` (*bypass*) | `false` | Baseline round-robin lexical + semântico sem fusão de scores |
| `e1_rr_todos.yaml` | `E1` | `tfidf`, `bm25`, `embedding` | `false` | `select_without_fusion` (*bypass*) | `false` | Round-robin com 3 recuperadores + **execução-mestre** que captura `retrieval` completo para replay offline |
| `e2_rrf_k10_bm25_emb.yaml` | `E2` | `bm25`, `embedding` | `true` | `strategy: rrf`, `rrf_k: 10` | `false` | Testar RRF exatamente no limiar crítico $k = 10$ ($1/11 == 2/22$ em $C=12$) |
| `e2_rrf_k60_bm25_emb.yaml` | `E2` | `bm25`, `embedding` | `true` | `strategy: rrf`, `rrf_k: 60` | `false` | Avaliar RRF padrão ($k=60$) contra round-robin (`e1_rr_bm25_emb`) |
| `e2_rrf_k60_todos.yaml` | `E2` | `tfidf`, `bm25`, `embedding` | `true` | `strategy: rrf`, `rrf_k: 60` | `false` | Medir o efeito do viés lexical 2:1 (`tfidf` + `bm25` vs. `embedding`) no RRF padrão |
| `e4_weighted_rrf_todos.yaml` | `E4` | `tfidf`, `bm25`, `embedding` | `true` | `strategy: weighted_rrf`, `weights: {tfidf: 1.0, bm25: 1.0, embedding: 2.0}` | `false` | Testar se dobrar o peso do `embedding` ($w_{\text{emb}}=2$) neutraliza o viés lexical 2:1 |
| `e4_convex_bm25_emb.yaml` | `E4` | `bm25`, `embedding` | `true` | `strategy: convex`, `weights: {bm25: 1.0, embedding: 1.0}` | `false` | Comparar fusão por magnitude Min-Max (`convex`) com RRF e medir efeito de *outliers* do BM25 |
| `e4_dbsf_bm25_emb.yaml` | `E4` | `bm25`, `embedding` | `true` | `strategy: dbsf`, `weights: {bm25: 1.0, embedding: 1.0}` | `false` | Avaliar normalização $\mu \pm 3\sigma$ (`dbsf`) e seu bônus de presença frente ao Min-Max |
| `e5_coupling_rr_rerank.yaml` | `E5` | `bm25`, `embedding` | `false` | `select_without_fusion` (*bypass*) | `true` | Par 1 de E5: medir `evidence` final com `hybrid: false` e `reranking: true` |
| `e5_coupling_rrf_rerank.yaml` | `E5` | `bm25`, `embedding` | `true` | `strategy: rrf`, `rrf_k: 60` | `true` | Par 2 de E5: comprovar que `reranking: true` produz `top_k` idêntico ao de `hybrid: false` (Acoplamento A1) |

- **Trechos reais de dois YAMLs representativos (`e2_rrf_k60_bm25_emb.yaml` e `e4_weighted_rrf_todos.yaml`):**

```yaml
# experiments/g7/configs/e2_rrf_k60_bm25_emb.yaml (linhas 20-42)
pipeline:
  top_k: 5
  candidate_k: 12
  rrf_k: 60
  expanded_queries: 3
  sentence_radius: 1
  max_context_chars: 12000
  max_sentence_chars: 1000

hybrid:
  strategy: rrf

modules:
  multi_query: {enabled: false}
  hyde: {enabled: false}
  decomposicao: {enabled: false}
  tfidf: {enabled: false}
  bm25: {enabled: true}
  embedding: {enabled: true}
  hybrid: {enabled: true}
  reranking: {enabled: false}
  sentence: {enabled: false}
  compressao: {enabled: false}
```

```yaml
# experiments/g7/configs/e4_weighted_rrf_todos.yaml (linhas 29-46)
hybrid:
  strategy: weighted_rrf
  weights:
    tfidf: 1.0
    bm25: 1.0
    embedding: 2.0

modules:
  multi_query: {enabled: false}
  hyde: {enabled: false}
  decomposicao: {enabled: false}
  tfidf: {enabled: true}
  bm25: {enabled: true}
  embedding: {enabled: true}
  hybrid: {enabled: true}
  reranking: {enabled: false}
  sentence: {enabled: false}
  compressao: {enabled: false}
```

#### 5.6.5 Evidências Exatas de Verificação da F8

- **Validação estrita dos 13 YAMLs via `rag-lab --config <arquivo> validate` (`13/13 EXIT=0`):**
  ```text
  e0_bm25.yaml: EXIT=0
  e0_embedding.yaml: EXIT=0
  e0_tfidf.yaml: EXIT=0
  e1_rr_bm25_emb.yaml: EXIT=0
  e1_rr_todos.yaml: EXIT=0
  e2_rrf_k10_bm25_emb.yaml: EXIT=0
  e2_rrf_k60_bm25_emb.yaml: EXIT=0
  e2_rrf_k60_todos.yaml: EXIT=0
  e4_convex_bm25_emb.yaml: EXIT=0
  e4_dbsf_bm25_emb.yaml: EXIT=0
  e4_weighted_rrf_todos.yaml: EXIT=0
  e5_coupling_rrf_rerank.yaml: EXIT=0
  e5_coupling_rr_rerank.yaml: EXIT=0
  ```
- **Saída real do script de verificação do corpus e do gabarito contra `split_document(..., 1000)`:**
  ```text
  === CORPUS (12 ARQUIVOS) ===
  01_arquitetura_geral.md: 2234 chars | 13 chunks | sha256=4c803007a13b
  02_autenticacao_identidade.md: 2066 chars | 12 chunks | sha256=f7ddef0b5595
  03_banco_vetorial_indexacao.md: 2039 chars | 11 chunks | sha256=b903f81353dd
  04_filas_mensageria.md: 1826 chars | 11 chunks | sha256=8012327594e2
  05_faturamento_cotas.md: 1844 chars | 11 chunks | sha256=f413ef7747cd
  06_seguranca_criptografia.md: 2096 chars | 11 chunks | sha256=7266eec7b719
  07_backup_recuperacao.md: 1901 chars | 11 chunks | sha256=56616bc55ebb
  08_auditoria_conformidade.md: 1941 chars | 11 chunks | sha256=71f80bf3aeb8
  09_observabilidade_telemetria.md: 1961 chars | 11 chunks | sha256=e41d011c5593
  10_slas_disponibilidade.md: 1958 chars | 11 chunks | sha256=f549c8cbf295
  11_deploy_configuracao.md: 2065 chars | 11 chunks | sha256=3ce0076a485b
  12_governanca_ingestao_dados.md: 2060 chars | 11 chunks | sha256=d29201085777
  Total de chunks no corpus: 135
  === GABARITO ===
  Perguntas por perfil: {'literal': 6, 'semantica': 6, 'composta': 6, 'sem_resposta': 6}
  Trechos relevantes verificados: 35/35 no arquivo e 35/35 em chunk unico
  SHA-256 inicial de perguntas_gabarito.yaml: 046841378bc39d6dd95fd2edefe37f3d91161afed85802925d6e932c68762e32
  ```

### 5.7 Auditoria Algorítmica, Sincronização com `upstream/develop` e Piloto Ao Vivo (Docker + Ollama)

#### 5.7.1 Sincronização com `upstream/develop` (`923cd03`) e Estratégia das Duas Branches

- **Novidades incorporadas do repositório oficial do professor (`upstream/develop`, commit `923cd03`):**
  1. **Classificador de rótulos na ingestão (`tagging_method` e `src/rag_lab/tagging.py`):** `Settings` passa a declarar `tagging_method: TaggingMethod = "llm"` (`Literal["llm", "terms"]`) com preenchimento padrão `TAGGING_METHOD_ENV = "${RAG_TAG_CLASSIFIER:-llm}"` em `load_settings()`. Durante a ingestão, cada chunk recebe `metadata["tags"]` a partir do catálogo `config/tags-si.yaml`.
  2. **Atualização do modelo de embeddings em `config/lab.yaml`:** o arquivo principal do laboratório (`config/lab.yaml`, mantido 100% intocado pelo G7) passou a referenciar `embeddinggemma:300m` (`dimensions: 768`) e `tagging_method: ${RAG_TAG_CLASSIFIER:-llm}`.
  3. **Novos testes oficiais:** `tests/test_tagging.py`, `tests/test_cli.py` e `tests/test_pipeline.py::test_tags_follow_chunk_id_through_fusion_and_context_steps[False|True]`.
- **Resolução limpa no rebase de `feat/g7-hybrid` sobre `upstream/develop` (`923cd03`):**
  - Em `src/rag_lab/config.py`, preservamos integralmente `TaggingMethod` e `TAGGING_METHOD_ENV` ao lado de `RETRIEVER_NAMES`, `RetrieverName`, `FusionStrategy`, `Weight` e `hybrid: HybridConfig = Field(default_factory=HybridConfig)`.
  - Em `tests/test_config.py`, como o fixture `config_data` lê `config/lab.yaml` via `yaml.safe_load` sem expandir `${RAG_TAG_CLASSIFIER:-llm}`, definimos explicitamente `config_data["tagging_method"] = "llm"` nos testes diretos de `Settings.model_validate` do G7 (`test_hybrid_config_defaults_and_valid_strategies` e `test_invalid_hybrid_config_rejected`), garantindo que a validação exercite genuinamente `HybridConfig`.
  - Todos os 13 YAMLs em `experiments/g7/configs/*.yaml` continuam 100% válidos sem necessidade de edição (`load_settings` injeta `tagging_method` automaticamente via `setdefault`).
- **Estratégia das duas branches e unificação de `_resolve_hybrid_config`:**
  - **`g7-hybrid` (Trilha A — entrega de arquivo único):** modifica exclusivamente `src/rag_lab/modules/hybrid.py` sobre `upstream/develop` (`923cd03`), sem depender de símbolos novos em `config.py`.
  - **`feat/g7-hybrid` (Trilha B — projeto completo F0–F11):** contém `HybridConfig` em `config.py`, a suíte completa de testes, o pacote experimental `experiments/g7/` e este documento vivo.
  - **Unificação do código de `src/rag_lab/modules/hybrid.py` entre ambas as branches:** `hybrid.py` agora é idêntico nas duas branches gracias ao resolvedor em três níveis `_resolve_hybrid_config(payload, settings)`:
    1. **Prioridade 1 (`settings.hybrid`):** se `getattr(settings, "hybrid", None)` existir (caso de `feat/g7-hybrid`), utiliza `settings.hybrid.strategy` e `settings.hybrid.weights`.
    2. **Prioridade 2 (`payload["attributes"]["hybrid_config"]`):** se `settings` não possuir `.hybrid` (caso de `upstream/develop` puro na branch `g7-hybrid` ou em stubs de teste), lê e valida estritamente `payload["attributes"]["hybrid_config"]` (`strategy` e `weights` com $0 \le w \le 100$, rejeitando `bool`, `NaN`/`Inf` e chaves desconhecidas).
    3. **Prioridade 3 (Padrão):** retorna `("rrf", {})`.

```python
def _resolve_hybrid_config(payload: dict, settings: Any) -> tuple[str, dict[str, Any]]:
    hybrid_cfg = getattr(settings, "hybrid", None)
    if hybrid_cfg is not None:
        return hybrid_cfg.strategy, hybrid_cfg.weights

    attributes = payload.get("attributes")
    if not isinstance(attributes, dict) or "hybrid_config" not in attributes:
        return "rrf", {}

    raw_cfg = attributes["hybrid_config"]
    if not isinstance(raw_cfg, dict):
        raise ValueError("attributes.hybrid_config deve ser um dicionário.")
    unknown_keys = set(raw_cfg) - {"strategy", "weights"}
    if unknown_keys:
        raise ValueError(
            f"Opções desconhecidas em attributes.hybrid_config: {', '.join(sorted(unknown_keys))}"
        )

    strategy = raw_cfg.get("strategy", "rrf")
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError(f"Estratégia híbrida desconhecida: {strategy}")

    raw_weights = raw_cfg.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("attributes.hybrid_config.weights deve ser um dicionário.")
    weights: dict[str, float] = {}
    for retriever, weight in raw_weights.items():
        if not isinstance(retriever, str) or not retriever:
            raise ValueError("Nome de recuperador inválido em attributes.hybrid_config.weights.")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 100
        ):
            raise ValueError(f"Peso inválido para o recuperador '{retriever}'.")
        weights[retriever] = float(weight)
    return strategy, weights
```

#### 5.7.2 Auditoria Algorítmica de `src/rag_lab/modules/hybrid.py`: Invariância de Escala no Desempate e `boundary_tie`

Durante a análise inicial do piloto ao vivo (`S02__rrf60.json`), surgiu a questão: por que, no empate de RRF em $\frac{1}{60 + 3} = \frac{1}{63} \approx 0{,}01587302$ entre o 3º colocado exclusivo do `bm25` (`07_backup_recuperacao.md`, pos 7, score bruto `9.9271`) e o 3º colocado exclusivo do `embedding` (`06_seguranca_criptografia.md`, pos 5, score bruto `0.6865`), o candidato do `bm25` ficou em `fused_rank = 6` e o gabarito do `embedding` ficou em `fused_rank = 7`?

1. **Verificação da chave de ordenação em `src/rag_lab/modules/hybrid.py` (linha 213):**
   ```python
   ordered = sorted(merged.values(), key=lambda e: (-e["score"], e["id"]))
   ```
   - **Diagnóstico:** **Não havia bug no código de `hybrid.py`** (houve apenas uma interpretação equivocada na prosa do relatório preliminar). O módulo `Hybrid` **nunca** comparou `raw_score` bruto nem `best_rank` para desempatar candidatos.
   - **Por que comparar `raw_score` entre recuperadores heterogêneos seria um erro matemático grave:** o `bm25` produz scores ilimitados em $(0, +\infty)$ (tipicamente `5.0` a `35.0+`), enquanto `tfidf` e `embedding` operam com similaridade de cosseno limitada a $\le 1{,}0$. Qualquer critério de desempate que utilizasse `max(raw_score)` introduziria um viés sistemático de 100% a favor do `bm25` em todos os empates de posição $(r_{\text{bm25}} = r_{\text{emb}})$ do RRF, destruindo a propriedade fundamental de invariância de escala do RRF.
2. **Prova empírica com os `chunk_id` SHA-256 reais de `data/results/teste_ao_vivo/S02__rrf60.json`:**
   - Em `S02__rrf60.json`, ocorreram quatro pares de empate exato entre candidatos exclusivos (`m = 1`) na mesma posição $r \in \{2, 3, 5, 6\}$:

| Posição $r$ na lista de origem | Score RRF (`1 / (60 + r)`) | Candidato `bm25` (`id` prefixo 12 hex, `raw_score`) | Candidato `embedding` (`id` prefixo 12 hex, `raw_score`) | Vencedor do desempate `(-score, id)` | `fused_rank` resultante |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **$r = 2$** | `1/62 = 0.01612903` | `2387f9009d87...` (`11_deploy` pos 2, `10.1456`) | `86cf26a2359e...` (`10_slas` pos 0, `0.7082`) | `"2387..." < "86cf..."` (`bm25`) | `#4` vs. `#5` *(ambos no Top-5)* |
| **$r = 3$** | `1/63 = 0.01587302` | `54833391d69b...` (`07_backup` pos 7, `9.9271`) | `e59f8e132b52...` (`06_seguranca` pos 5, `0.6865`) **[ALVO]** | `"5483..." < "e59f..."` (`bm25`) | `#6` vs. `#7` *(ambos fora do Top-5)* |
| **$r = 5$** | `1/65 = 0.01538462` | `df6be78ee17c...` (`09_observabilidade` pos 3, `8.9502`) | `909b83e155a1...` (`06_seguranca` pos 2, `0.6790`) | **`"909b..." < "df6b..."` (`embedding`)** | **`#9` (`emb`) vs. `#10` (`bm25`)** |
| **$r = 6$** | `1/66 = 0.01515152` | `424a91ce6ffb...` (`06_seguranca` pos 10, `8.5358`) | `107492924e6b...` (`06_seguranca` pos 0, `0.6720`) | **`"1074..." < "424a..."` (`embedding`)** | **`#11` (`emb`) vs. `#12` (`bm25`)** |

   - Como demonstram os pares $r = 5$ e $r = 6$, o candidato do `embedding` (com score bruto `~0.67`) venceu o candidato do `bm25` (com score bruto `~8.5–8.9`) sempre que seu hash SHA-256 foi lexicograficamente menor. Além disso, em $r = 3$, ambos os candidatos empatados (`#6` e `#7`) já estavam fora do corte `top_k = 5`, pois as 5 primeiras vagas foram ocupadas pelos 2 itens de consenso (`#1` e `#2`), pelo `#1` do `bm25` (`#3`) e pelos dois empatados em $r = 2$ (`#4` e `#5`).
3. **Auditoria de `boundary_tie` na fronteira `top_k` (linhas 232–237 de `hybrid.py`):**
   ```python
   boundary_tie = len(ordered) > top_k and math.isclose(
       ordered[top_k - 1]["score"],
       ordered[top_k]["score"],
       rel_tol=1e-12,
       abs_tol=1e-12,
   )
   ```
   - `boundary_tie` compara especificamente o último item **incluído** no contexto (`ordered[top_k - 1]`, posição `top_k` 1-based) com o primeiro item **excluído** do contexto (`ordered[top_k]`, posição `top_k + 1` 1-based).
   - Em `S02__rrf60.json` com `top_k = 5`, `ordered[4]` (`fused_rank = 5`) tem score $\frac{1}{62} \approx 0{,}01612903$ e `ordered[5]` (`fused_rank = 6`) tem score $\frac{1}{63} \approx 0{,}01587302$ (diferença $0{,}000256 > 10^{-12}$), razão pela qual `boundary_tie` é `false` em `top_k = 5` (os empates ocorreram nas fronteiras `4|5` e `6|7`, e não na fronteira de corte `5|6`). Se `top_k` fosse `4` ou `6`, `boundary_tie` seria `true`.
4. **Novos testes de regressão adicionados em `tests/test_hybrid.py` (elevando a suíte para 19 funções / 39 casos parametrizados):**
   - `test_rrf_tie_breaking_is_strictly_scale_invariant_by_chunk_id`: comprova que um chunk exclusivo do `embedding` em rank 3 (`score = 0.68`, `id = "a_emb_rank3"`) vence o desempate em $1/63$ contra um chunk exclusivo do `bm25` em rank 3 (`score = 30.0`, `id = "z_bm25_rank3"`).
   - `test_hybrid_preserves_metadata_tags_across_all_strategies`: parametrizado nas 4 estratégias (`rrf`, `weighted_rrf`, `convex`, `dbsf`), comprova que `metadata["tags"]` sobrevive intacto à fusão.
   - `test_hybrid_config_attribute_fallback_precedence_and_validation`: comprova a precedência `settings.hybrid > attributes["hybrid_config"] > ("rrf", {})` e a rejeição com `ValueError` de 9 formatos inválidos em `attributes["hybrid_config"]`.
   - `test_boundary_tie_true_and_false_at_top_k_cutoff`: reproduz a geometria exata de `S02`, provando `boundary_tie == False` para `top_k = 5` e `boundary_tie == True` para `top_k = 4` e `top_k = 6`.

#### 5.7.3 Registro do Piloto Ao Vivo no Docker Compose + Ollama (`L02` e `S02` nos 5 Modos)

- **Ambiente de execução do piloto (`data/results/teste_ao_vivo/`):**
  - Containers Docker Compose: `rag-lab-postgres-1` (`PostgreSQL 16; pgvector 0.8.7`, `Healthy`) e `rag-lab-app-1` (`Python 3.12.16`, `Healthy`).
  - Modelos Ollama locais validados via `rag-lab doctor`: `llm.model = lfm3:1.7b-q4_K_M` (`temperature: 0.0`) e `embeddings.model = nomic-embed-text:latest` (`dimensions: 768`).
  - Ingestão do corpus "Plataforma Núcleo Aurora" (`RAG_TAG_CLASSIFIER=terms`): **12 documentos `.md`** e **135 chunks** indexados em **~28 segundos**.
- **Tabela Resumo — Pergunta Literal `L02` (`ERR-AUR-417`, alvo: `03_banco_vetorial_indexacao.md` pos 4):**
  - Recuperadores isolados (`candidate_k = 12`): `bm25` trouxe o alvo em **#1** com *outlier* extremo (`score = 31.0986` vs. `15.0719` do 2º colocado; `raw_mean = 10.9018, raw_std = 6.8623`); `embedding` também trouxe o alvo em **#1** (`score = 0.8491`). Sobreposição: `19` candidatos únicos, `in_all = 5` (Jaccard $= 5/19 = 0{,}2632$).

| Modo | `#1` (`[1]`) | `#2` (`[2]`) | `#3` (`[3]`) | `#4` (`[4]`) | `#5` (`[5]`) | Rank do Alvo | Fenômeno Experimental Observado |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **`bypass_rr`** (`hybrid: false`) | **`03_banco (pos 4)`** (`31.0986`) | `03_banco (pos 1)` (`15.0719`) | `01_arquitetura (pos 11)` (`0.7580`, *só emb*) | `04_filas (pos 4)` (`14.4746`, *só bm25*) | `04_filas (pos 6)` (`0.7219`) | **`#1`** | Mistura escalas brutas (`31.09` vs. `0.75`) e promove itens sem consenso a `[3]` e `[4]` |
| **`rrf60`** ($k=60$, 1:1) | **`03_banco (pos 4)`** (`0.032787`) | `03_banco (pos 1)` (`0.031754`) | `04_filas (pos 6)` (`0.030366`) | `05_faturamento (pos 4)` (`0.030331`) | `03_banco (pos 5)` (`0.029670`) | **`#1`** | Teorema $k=60 > 10$: os 5 itens de consenso (`in_all: 5`) ocupam 100% do Top-5 |
| **`weighted_rrf`** (`1:2`) | **`03_banco (pos 4)`** (`0.049180`) | `03_banco (pos 1)` (`0.047379`) | `04_filas (pos 6)` (`0.046239`) | **`03_banco (pos 5)`** (`0.045055`) | `05_faturamento (pos 4)` (`0.045037`) | **`#1`** | Peso `2.0` no `embedding` promove `03_banco (pos 5)` acima do distrator `ERR-AUR-429` |
| **`convex`** (Min-Max, 1:1) | **`03_banco (pos 4)`** (`1.000000`) | `03_banco (pos 1)` (`0.336586`) | **`01_arquitetura (pos 11)`** (`0.252403`, *só emb*) | `05_faturamento (pos 4)` (`0.200304`) | `04_filas (pos 6)` (`0.183309`) | **`#1`** | **Achatamento por *outlier* no Min-Max:** `31.0986` esmaga o restante do BM25 e deixa `emb#2` isolado saltar para `[3]` |
| **`dbsf`** ($\mu \pm 3\sigma$, 1:1) | **`03_banco (pos 4)`** (`0.981098`) | `03_banco (pos 1)` (`0.571674`) | `05_faturamento (pos 4)` (`0.487403`) | `04_filas (pos 6)` (`0.477470`) | `03_banco (pos 5)` (`0.451225`) | **`#1`** | Resiste ao *outlier* do BM25 e mantém todos os 5 itens de consenso no Top-5 graças ao bônus de presença |

- **Tabela Resumo — Pergunta Semântica `S02` (sanitização HTTP 503 na borda, alvo: `06_seguranca_criptografia.md` pos 5):**
  - Recuperadores isolados (`candidate_k = 12`): `bm25` **não encontrou o alvo** (`Target rank: None`, atraído por distratores lexicais de `"endereço"`, `"servidor web"` e `"comunicação"`); `embedding` encontrou o alvo em **#3** (`score = 0.6865`). Sobreposição: `22` candidatos únicos, apenas `in_all = 2` (Jaccard $= 2/22 = 0{,}0909$).

| Modo | `#1` (`[1]`) | `#2` (`[2]`) | `#3` (`[3]`) | `#4` (`[4]`) | `#5` (`[5]`) | Rank do Alvo (`06_seguranca pos 5`) | Comportamento do LLM (`lfm3:1.7b-q4_K_M`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **`bypass_rr`** (`hybrid: false`) | `03_banco (pos 3)` (`bm25#1`) | `01_arquitetura (pos 2)` (`emb#1+bm25#4`) | `11_deploy (pos 2)` (`bm25#2`) | `10_slas (pos 0)` (`emb#2`) | `07_backup (pos 7)` (`bm25#3`) | **6º lugar** *(cortado)* | **Alucinação por distrator:** `top_k=5` ímpar dá 3 vagas ao `bm25` e corta `emb#3`; LLM atribui proteção ao hash SHA-256 `[3]` |
| **`rrf60`** ($k=60$, 1:1) | `01_arquitetura (pos 2)` (`0.032018`) | `01_arquitetura (pos 5)` (`0.029631`, `bm25#8+emb#7`) | `03_banco (pos 3)` (`0.016393`, `bm25#1`) | `11_deploy (pos 2)` (`0.016129`, `bm25#2`) | `10_slas (pos 0)` (`0.016129`, `emb#2`) | **7º lugar** (`0.015873`) | ***Fusion Tax* clássico:** consenso mediano (`#8+#7`) + `bm25#1` e `bm25#2` expulsam `emb#3` do Top-5; LLM improvisa sem a fonte |
| **`convex`** (Min-Max, 1:1) | `01_arquitetura (pos 2)` (`0.762708`) | `03_banco (pos 3)` (`0.500000`, `bm25#1`) | `11_deploy (pos 2)` (`0.487607`, `bm25#2`) | `07_backup (pos 7)` (`0.442382`, `bm25#3`) | `10_slas (pos 0)` (`0.410707`, `emb#2`) | **7º lugar** (`0.239950`) | Topo plano do BM25 (`10.21, 10.15, 9.93`) domina `[2]..[4]` e mantém o gabarito fora do Top-5 |
| **`dbsf`** ($\mu \pm 3\sigma$, 1:1) | `01_arquitetura (pos 2)` (`0.718424`) | `01_arquitetura (pos 5)` (`0.399584`) | `03_banco (pos 3)` (`0.390783`, `bm25#1`) | `10_slas (pos 0)` (`0.389140`, `emb#2`) | `11_deploy (pos 2)` (`0.384926`, `bm25#2`) | **7º lugar** (`0.292304`) | Fusão simétrica 1:1 mantém os 2 consensos + `bm25#1`, `emb#2`, `bm25#2` no Top-5 |
| **`weighted_rrf`** (`bm25=1, emb=2`) | `01_arquitetura (pos 2)` (`0.048412`) | `01_arquitetura (pos 5)` (`0.044557`) | `10_slas (pos 0)` (`0.032258`, `emb#2`) | **`06_seguranca (pos 5)`** (`0.031746`, `emb#3`) | `01_arquitetura (pos 12)` (`0.031250`, `emb#4`) | **4º lugar (`[4]`)** | **Resgate completo:** $\frac{2{,}0}{63} = 0{,}031746 > \frac{1{,}0}{61} = 0{,}016393$; LLM responde exatamente o erro HTTP 503 sem exibir URLs/chaves e cita `[4]` |

- **Reavaliação offline pós-auditoria:** a reexecução offline do `Hybrid.run` unificado sobre os objetos `retrieval` salvos em `L02__rrf60.json` e `S02__rrf60.json` confirmou `identical_ranking = True` em ambos os casos (`target_fused_rank = 1` em `L02` e `target_fused_rank = 7` em `S02`, `boundary_tie = False`).

### 5.8 Fase F9 — Ferramental de Coleta (`rodar.py`), Replay Offline Exato (`avaliar.py`) e Testes (`tests/test_g7_avaliacao.py`)

- **O que foi construído:**
  1. **`experiments/g7/rodar.py` (392 linhas):**
     - CLI completo (`--config`, `--gabarito`, `--corpus-dir`, `--out-dir`, `--run-id`, `--reps`, `--profiles`, `--ids`, `--llm-model`, `--embedding-model`, `--dry-run`) que executa `Pipeline(services).run(...)` em processo.
     - **`InstrumentedProviders`:** wrapper sobre `Providers` que contabiliza por consulta `chat_calls`, `embed_calls` e `embed_texts` sem modificar o `Envelope` retornado pelo `Pipeline`.
     - **Sanitização estrita de segredos em `manifest.json` (`sanitize_settings` e `_redact_url`):** exclui `database_url` (registrando apenas `database_name`, ex.: `"rag_g7"`), remove qualquer usuário/senha de `base_url` e preserva apenas o nome simbólico em `api_key_env`.
     - **Regra determinística de hash multiplataforma (`hash_mode: "sha256-lf"`):**
       - No Windows, `git rebase` com `core.autocrlf=true` converte arquivos de trabalho de `LF` (`\n`) para `CRLF` (`\r\n`), o que alteraria o SHA-256 bruto de `experiments/g7/perguntas_gabarito.yaml` de `04684137...` (calculado em `LF` na F8) para `6cc7fada...` (em `CRLF`).
       - Para garantir reprodutibilidade 100% determinística entre Windows (host) e Linux (Docker/CI), `file_sha256(path, normalize_lf=True)` normaliza `raw.replace(b"\r\n", b"\n")` antes de calcular o digest e grava `"hash_mode": "sha256-lf"` no `manifest.json`, restaurando o hash canônico `046841378bc39d6dd95fd2edefe37f3d91161afed85802925d6e932c68762e32` do gabarito e os hashes originais dos 12 arquivos `.md` do corpus.
     - **Rastreabilidade completa (`g7-run-manifest/1`):** grava commit Git, branch, `git_dirty`, ambiente Python/OS/CPU, `config_sha256`, `uv_lock_sha256`, `gabarito_sha256`, `gabarito_homologado` (`bool` e `gabarito_status_counts`), mapa `corpus_sha256` dos 12 arquivos `.md` e metadados `/api/show` dos modelos Ollama.
  2. **`experiments/g7/avaliar.py` (777 linhas):**
     - **Compatibilidade dupla de leitura (`load_run_records`):** aceita tanto diretórios `<run_id>/envelopes/*.json` gerados por `rodar.py` quanto diretórios com envelopes `Envelope` diretos (como `data/results/teste_ao_vivo/*.json`).
     - **Simetria metodológica na reconstrução do ranking completo (`extract_full_ranking`):**
       - Como `build_context` trunca `payload["evidence"]` em `top_k = 5`, avaliar `Recall@10` e `MRR` apenas sobre `envelope["evidence"]` em `E0` e `E1` penalizaria injustamente as baselines sem `attributes.hybrid`. Por isso, `extract_full_ranking(envelope)` reconstrói a lista pós-seleção completa em todos os blocos:
         - **Bloco E0 (`hybrid: false`, 1 recuperador ativo):** usa `retrieval[unico_recuperador]` (até `candidate_k = 12` candidatos).
         - **Bloco E1 / `bypass_rr` (`hybrid: false`, 2 ou 3 recuperadores ativos):** reconstrói a lista intercalada completa via `select_without_fusion(retrieval)` (até 24–36 candidatos únicos).
         - **Blocos E2 e E4 (`hybrid: true`, `reranking: false`):** usa `attributes.hybrid.ranking` (com texto recuperado via `build_text_lookup(envelope)`).
         - **Bloco E5 (`reranking: true`):** usa `envelope["evidence"]` reordenado pelo `Reranking`.
       - Assim, `Hit@5`, `Recall@5` e `nDCG@5` refletem estritamente a janela `top_k = 5` enviada ao LLM, enquanto `Recall@10` e `MRR` operam com simetria total sobre o ranking completo em `E0`, `E1`, `E2` e `E4`.
     - **Fórmulas matemáticas exatas implementadas para `MRR` e `nDCG@5` (`compute_document_metrics` e `compute_snippet_metrics`):**
       - **`MRR` (sobre a lista completa reconstruída $L$):**
         $$\text{MRR} = \frac{1}{r^*}, \quad r^* = \min \{r \in \{1, \dots, |L|\} \mid L[r-1] \text{ é relevante}\} \quad (\text{ou } 0{,}0 \text{ se ausente})$$
       - **`nDCG@5` graduado por `essencial` e sem ganho duplicado:**
         - Para documentos (`doc_ndcg_5`), cada documento relevante $d \in D_{\text{rel}}$ tem ganho binário $g(d) = 1{,}0$ creditado **apenas na sua primeira aparição** $r \le 5$.
         - Para trechos (`trecho_ndcg_5`), cada trecho do gabarito $t_m \in T_{\text{rel}}$ possui ganho graduado $g(t_m) = 2{,}0$ se `essencial: true` e $g(t_m) = 1{,}0$ se `essencial: false`, creditado **apenas na primeira posição** $r \le 5$ cujo chunk contenha a substring normalizada de $t_m$:
           $$\text{DCG@5} = \sum_{r=1}^{5} \frac{\text{gain}(r)}{\log_2(r + 1)}, \qquad \text{IDCG@5} = \sum_{r=1}^{\min(5, |T_{\text{rel}}|)} \frac{g_{(r)}}{\log_2(r + 1)}, \qquad \text{nDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}}$$
           onde $g_{(1)} \ge g_{(2)} \ge \dots$ são os ganhos de $T_{\text{rel}}$ ordenados de forma decrescente.
       - Perguntas `sem_resposta` (`responder: false`) são excluídas das médias de recuperação e avaliadas separadamente na taxa de abstenção.
     - **Motor de *Replay* Offline Exato (`replay_from_retrieval` e `verify_replay_equivalence`):**
       - Reexecuta os próprios `select_without_fusion` (`src/rag_lab/pipeline.py`) e `Hybrid().run` (`src/rag_lab/modules/hybrid.py`) em memória sobre `payload["retrieval"]` para varrer `rrf_k` $\in \{1, 3, 5, 10, 11, 15, 21, 22, 30, 60, 120, 1000\}$ (**E2**), todas as permutações de ordem do *round-robin* (**E3**) e as 4 estratégias de fusão (**E4**).
       - Verifica automaticamente `replay_ids == online_ids` contra todos os envelopes online salvos (`replay_verificacao.json`).
     - **Detector de *Fusion Tax* e Classificação de Severidade na Tabela `T5` (`detect_fusion_tax_for_envelope`):**
       - Avalia **exclusivamente** execuções com `hybrid` executado (`status == "executed"` e `len(retrieval) >= 2`), garantindo que `E0` e `E1`/`bypass_rr` jamais sejam listados como *Fusion Tax*.
       - Classifica cada ocorrência na coluna `severidade`:
         - `"perda_top5"` (`fused_recall_5 < max(best_single_recall_5, rr_recall_5)`): degradação severa em que a fusão expulsa um trecho relevante para fora da janela `top_k = 5` enviada ao LLM (caso de `rrf60`, `convex` e `dbsf` em `S02`: rank `#3` no `embedding` -> rank `#7` na fusão).
         - `"queda_mrr_no_top5"` (`fused_recall_5 >= ref_recall_5 > 0` e `fused_mrr < ref_mrr`): o trecho relevante permanece dentro do `top_k = 5` (o LLM recebe a evidência e responde corretamente), mas em posição ligeiramente inferior à do melhor recuperador isolado (caso de `weighted_rrf` em `S02`: rank `#3` no `embedding` -> rank `#4` na fusão, resgatando a resposta do LLM).
         - `"queda_mrr_fora_top5"`: quando o trecho já estava fora do `top_k = 5` na referência e sofreu queda adicional de posição na cauda.
     - **Geração automática das 6 tabelas (`.csv` e `.md`) + Template de Avaliação Humana:**
       - `t1_recuperacao.{csv,md}`, `t2_curva_rrf_k.{csv,md}`, `t3_geracao_citacoes.{csv,md}`, `t4_latencia_chamadas.{csv,md}`, `t5_fusion_tax.{csv,md}`, `t6_sobreposicao_empates.{csv,md}` e `avaliacao_humana_template.csv`.
  3. **`tests/test_g7_avaliacao.py` (415 linhas, 8 funções / 13 casos parametrizados):**
     - Testa valores manuais exatos de `Recall@{1,3,5,10}`, `Hit@5`, `MRR`, `nDCG@5` (documento e trecho, deduplicação e pesos `essencial`), reconstrução simétrica de ranking completo além do `top_k = 5` para `E0` e `E1` (`rank #6` contabilizado em `Recall@10 = 1.0` e `MRR = 1/6`), `jaccard_index`, parser e precisão de citações `[n]`, detecção de abstenção, detector de *Fusion Tax* com severidades `perda_top5` vs. `queda_mrr_no_top5` e exclusão de `E0`/`E1`, equivalência `replay == online` nas 4 estratégias e em `round_robin`, determinismo `sha256-lf` (`LF == CRLF`), sanitização de segredos em `manifest.json` e fluxo ponta-a-ponta `rodar.py` -> `avaliar.py`.
- **Resumo de Comandos CLI (`rodar.py` e `avaliar.py`):**
  ```powershell
  # 1. Validação dry-run de configuração, gabarito e hashes sha256-lf (sem banco/Ollama):
  uv run --no-sync python experiments/g7/rodar.py --config experiments/g7/configs/e2_rrf_k60_bm25_emb.yaml --dry-run

  # 2. Coleta controlada online (com filtro opcional de IDs/perfis e override opcional de modelos):
  uv run --no-sync python experiments/g7/rodar.py --config experiments/g7/configs/e1_rr_todos.yaml --run-id e1_rr_todos
  uv run --no-sync python experiments/g7/rodar.py --config experiments/g7/configs/e2_rrf_k60_bm25_emb.yaml --ids L02 S02 --llm-model lfm3:1.7b-q4_K_M --embedding-model nomic-embed-text:latest

  # 3. Avaliação completa (gera T1-T6 em .csv/.md, replay_verificacao.json e avaliacao_humana_template.csv):
  uv run --no-sync python experiments/g7/avaliar.py --runs-dir data/results/g7 --out-dir experiments/g7/results
  uv run --no-sync python experiments/g7/avaliar.py --run-dir data/results/teste_ao_vivo --out-dir data/results/teste_ao_vivo/avaliacao_piloto
  ```
- **Descoberta experimental adicional no Replay Offline do Piloto (`t2_curva_rrf_k.md`, bloco E3):**
  - Na avaliação dos 10 envelopes reais do piloto (`L02` e `S02`), `verify_replay_equivalence` confirmou `10/10` envelopes idênticos (`all_matched: true`), o detector de *Fusion Tax* (`t5_fusion_tax.md`) classificou `rrf60`, `convex` e `dbsf` como `perda_top5` e `weighted_rrf` como `queda_mrr_no_top5` em `S02`, e a permutação de ordem do *round-robin* (**E3**) demonstrou que inverter `rr(bm25->embedding)` para `rr(embedding->bm25)` eleva o `trecho_recall_5` do piloto de `0.5000` para `1.0000` (e o `trecho_mrr` de `0.5833` para `0.6000`), comprovando quantitativamente o viés de ordem do `MODULE_ORDER` quando `top_k = 5` é ímpar.

---

## 6. Especificação Técnica das Próximas Fases (F3 a F11)

### 6.1 Fases F3 a F6 — Arquitetura Interna de `src/rag_lab/modules/hybrid.py` e Testes Unitários

#### Blueprint de `src/rag_lab/modules/hybrid.py`

Usaremos apenas a biblioteca padrão do Python (`copy.deepcopy`, `math.fsum`, `math. isclose`, `math.isfinite`, `statistics.fmean`, `statistics.pstdev`, `collections.abc.Callable`), mantendo `uv.lock` inalterado:

```text
+-----------------------------------------------------------------------+
| Hybrid.run(payload, services)                                         |
|   1. Lê strategy, configured_weights, rrf_k, top_k de services.settings|
|   2. Extrai retrievers = list(payload["retrieval"].keys())            |
|   3. Resolve pesos efetivos (_resolve_weights):                       |
|      - Valida strategy vs configured_weights e set(weights)==set(R)   |
|      - Se weights vazio: {r: 1.0 for r in retrievers}                 |
|      - Calcula weight_sum = math.fsum(weights.values())               |
|   4. Para cada recuperador r e sua lista ranking:                     |
|      - Extrai e valida raw_scores (_extract_raw_scores):              |
|        * Verifica math.isfinite(score); se não, levanta ValueError    |
|      - Calcula estatísticas descritivas (min, max, mean, pstdev)      |
|      - Chama STRATEGIES[strategy](raw_scores, weight, weight_sum, k)  |
|        -> devolve list[float] de contribuições já escaladas           |
|   5. Acumula por item["id"]:                                          |
|      - deepcopy da 1ª ocorrência, merge de item["scores"]             |
|      - ranks_by_id[id][r] = rank (1-based)                            |
|      - contribs_by_id[id][r] = contribution                           |
|   6. Para cada documento único:                                       |
|      - fused = math.fsum(contribs_by_id[id].values())                 |
|      - item["score"] = fused                                          |
|   7. Ordena payload["evidence"] por (-item["score"], item["id"])      |
|   8. Atribui item["scores"]["hybrid"] = item["score"]                 |
|   9. Constrói e grava payload["attributes"]["hybrid"]                 |
+-----------------------------------------------------------------------+
```

- **Assinatura das funções de estratégia (`Scorer`):**
  ```python
  Scorer = Callable[[list[float], float, float, int], list[float]]
  # Argumentos: (raw_scores, weight, weight_sum, rrf_k) -> list[float] (contribuições por posição)
  ```
  - `_score_rrf(raw_scores, weight, weight_sum, rrf_k)`:
    `[1.0 / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]`
  - `_score_weighted_rrf(raw_scores, weight, weight_sum, rrf_k)`:
    `[weight / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]`
  - `_score_convex(raw_scores, weight, weight_sum, rrf_k)`:
    Se `not raw_scores`: retorna `[]`. Calcula `lo = min(raw_scores)`, `hi = max(raw_scores)` e `factor = weight / weight_sum`. Se `hi == lo`: retorna `[factor for _ in raw_scores]`. Caso contrário: `[factor * ((s - lo) / (hi - lo)) for s in raw_scores]`.
  - `_score_dbsf(raw_scores, weight, weight_sum, rrf_k)`:
    Se `not raw_scores`: retorna `[]`. Calcula `mean = fmean(raw_scores)`, `std = pstdev(raw_scores)` e `factor = weight / weight_sum`. Se `std == 0.0`: retorna `[factor for _ in raw_scores]`. Caso contrário: `lo = mean - 3.0 * std`, `span = 6.0 * std`, e para cada `s` calcula `norm = min(1.0, max(0.0, (s - lo) / span))` retornando `factor * norm`.

- **Extração e validação do score bruto (`_extract_raw_scores`):**
  - Para cada `item` na lista `payload["retrieval"][retriever]`, lê `raw = item.get("scores", {}).get(retriever, item.get("score", 0.0))`.
  - Verifica `if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw): raise ValueError(...)`.

- **Divisão incremental entre F3, F4, F5 e F6:**
  - **F3:** Estrutura `STRATEGIES`, `_score_rrf`, validação de `weights` e scores não finitos, `math.fsum`, construção completa de `payload["attributes"]["hybrid"]` e testes em `tests/test_hybrid.py` cobrindo telemetria, `boundary_tie`, `fsum` invariante à ordem das chaves, sobrescrita de `attributes.hybrid` injetado e erro em `NaN`/`Inf`.
  - **F4:** Registro de `_score_weighted_rrf` em `STRATEGIES` e testes de equivalência com pesos `1.0`, inversão do viés 2:1 com `w_embedding = 2.0`, peso `0.0` mantendo candidato com contribuição zero e erro quando `weights` em runtime não cobre `payload["retrieval"]`.
  - **F5:** Registro de `_score_convex` em `STRATEGIES` e testes de invariância à escala linear (`bm25 * 1000`), sensibilidade a *outlier* do BM25, casos degenerados ($\max = \min$ e lista de 1 item) e fechamento `math.fsum(contributions.values()) == fused` inclusive quando um recuperador ativo retorna `[]`.
  - **F6:** Registro de `_score_dbsf` em `STRATEGIES` e testes de $\sigma = 0$, *clip* de *outlier* em lista com $n = 12$ itens, bônus de presença para o último item da lista ($\hat{s} > 0$) e contrato JSON/Envelope (`json.dumps(..., allow_nan=False)` + `Envelope.model_validate`) parametrizado para todas as 4 estratégias.

### 6.2 Fase F7 — Testes de Composição em `tests/test_pipeline.py`

Adicionar em `tests/test_pipeline.py`:
1. `test_pipeline_hybrid_strategies_emit_telemetry_and_preserve_contract`: parametrizado para `rrf`, `weighted_rrf`, `convex` e `dbsf` com `bm25` + `embedding` + `hybrid` ativos; verifica `status == "executed"` no `trace`, presença de `result.attributes["hybrid"]["strategy"] == strategy`, `len(result.attributes["hybrid"]["ranking"]) >= len(result.evidence)` e serialização `result.model_dump_json()`.
2. `test_pipeline_without_hybrid_uses_round_robin_and_leaves_hybrid_bypassed`: com `tfidf`, `bm25` e `embedding` ativos e `hybrid` desativado; verifica `status == "bypass"` para `hybrid`, ausência de `"hybrid"` em `result.attributes` e em `result.evidence[0].scores`.

### 6.3 Fase F8 — Corpus, Perguntas e Gabarito Experimental (`experiments/g7/`)

- **Corpus (`experiments/g7/corpus/`):** 10 a 15 documentos `.md` em português (1.000 a 3.000 caracteres cada), autorais/didáticos do grupo G7 sobre arquitetura de sistemas de recuperação de informação e RAG, construídos com:
  - Termos técnicos literais, siglas e códigos de configuração (favorecem `tfidf`/`bm25`);
  - Conceitos explicados por paráfrases e sinônimos sem repetir o jargão da pergunta (favorecem `embedding`);
  - Pares de documentos com vocabulário sobreposto mas semântica distinta (armadilhas lexicais para induzir e medir *Fusion Tax*);
  - Fatos complementares distribuídos em documentos distintos (para perguntas compostas).
- **Conjunto fixo de 24 perguntas (`experiments/g7/gabarito.yaml`):**
  - 6 **literais** (`L01`–`L06`), 6 **semânticas** (`S01`–`S06`), 6 **compostas** (`C01`–`C06`) e 6 **sem resposta na base** (`N01`–`N06`, sendo 3 *near-miss* com termos presentes no corpus e 3 fora de domínio).
- **Protocolo de gabarito independente:** dupla anotação manual (Diego e Gabriel Alves, com adjudicação por João Victor Póvoa) **antes** de rodar o sistema, registrando `documentos_relevantes`, `trechos_relevantes` (por substring normalizada, independente de hash/posição) e `fatos_esperados`. O SHA-256 de `gabarito.yaml` é gravado nos manifestos das execuções.

### 6.4 Fase F9 — Ferramental de Coleta, Replay Offline e Avaliação (`experiments/g7/`)

- **`experiments/g7/rodar.py`:**
  - Executa `Pipeline(services).run(...)` em processo para uma configuração YAML e salva cada envelope em `data/results/g7/<run_id>/envelopes/<id_pergunta>__rep<r>.json`.
  - Envolve `services.providers` com um contador de chamadas (`chat_calls`, `embed_calls`, `embed_texts`) e registra `manifest.json` completo (commit Git, status da working tree, SHA-256 de `uv.lock`, do YAML, do gabarito e de cada arquivo do corpus, versão do Python, hardware/OS, data ISO-8601 e digests dos modelos no Ollama).
- **`experiments/g7/avaliar.py`:**
  - **Motor de *Replay* Offline Exato:** lê os envelopes de uma execução com `tfidf + bm25 + embedding` ativos (`payload["retrieval"]` completo) e reexecuta `select_without_fusion` e `Hybrid().run` em memória para qualquer subconjunto de recuperadores, qualquer permutação de ordem no *round-robin*, qualquer `rrf_k` e qualquer estratégia/peso.
  - **Verificação de consistência:** compara automaticamente o ranking do *replay* offline contra os envelopes das execuções online correspondentes (`assert replay_ids == online_ids`).
  - **Tabelas geradas (`experiments/g7/results/`):**
    - **T1:** `Recall@{1,3,5,10}` (por documento e por trecho), `Hit@5`, `MRR` e `nDCG@5` por configuração e por perfil de pergunta.
    - **T2:** Curva de sensibilidade de `rrf_k` $\in \{1, 3, 5, 10, 11, 15, 21, 22, 30, 60, 120, 1000\}$ destacando os regimes $k \le 10$, $10 < k \le 21$ e $k > 21$.
    - **T3:** Métricas de geração: validade sintática das citações `[n]`, precisão das fontes citadas frente ao gabarito, cobertura de fatos esperados, taxa de abstenção correta em `N01`–`N06` e espaço para nota manual de fidelidade (rubrica 0–2).
    - **T4:** Latência por etapa (`trace[].duration_ms` mediano e p95) e contagem de chamadas aos modelos.
    - **T5:** Casos concretos de ***Fusion Tax*** (perguntas em que a fusão obteve MRR ou Recall@5 inferior ao melhor recuperador individual ou ao *round-robin*, com decomposição de ranks e contribuições).
    - **T6:** Sobreposição entre recuperadores (Jaccard entre top-12 de `tfidf`, `bm25` e `embedding`) e incidência de `boundary_tie`.
- **`tests/test_g7_avaliacao.py`:** testes unitários das funções de `Recall@k`, `MRR`, `nDCG@k`, validação de citações e consistência do motor de *replay*.

### 6.5 Fase F10 — Execução da Matriz Experimental (E0 a E6)

1. **Setup do banco dedicado (`rag_g7`):**
   - Criar `compose.override.yaml` local (não versionado) expondo `127.0.0.1:5432:5432` no serviço `postgres`.
   - Criar o banco `rag_g7` com `CREATE EXTENSION IF NOT EXISTS vector;` e ingerir apenas `experiments/g7/corpus/` com `embedding: true` (sem misturar `examples/documents/rag.txt`).
2. **Execução controlada:**
   - **E0:** Baselines individuais (`tfidf`, `bm25`, `embedding`).
   - **E1:** *Round-robin* (`hybrid: false`, confirmando `status: "bypass"` no `trace`).
   - **E2:** Varredura de `rrf_k` (replay completo + online em $k \in \{\text{melhor}, 60\}$).
   - **E3:** Quantificação do viés de ordem do *round-robin* nas 6 permutações de `(tfidf, bm25, embedding)`.
   - **E4:** Comparação das estratégias `rrf`, `weighted_rrf`, `convex` e `dbsf`.
   - **E5:** Demonstração empírica do acoplamento com `reranking: true` (`hybrid: true` vs. `hybrid: false` produzindo `evidence` final idêntica).
   - **E6:** Repetições online para medir estabilidade da geração no LLM.

### 6.6 Fase F11 — Relatório Final (`docs/g7-hybrid.md`) e Pull Request

- Redigir `docs/g7-hybrid.md` consolidando a pergunta de pesquisa, protocolo experimental, tabelas T1–T6, análise crítica dos casos de *Fusion Tax*, limitações estruturais da fusão e recomendação fundamentada de quando manter ou rejeitar cada estratégia.
- Rodar toda a verificação de CI (`ruff check .`, `rag-lab validate`, `rag-lab doctor`, `pytest` completo incluindo `test_api.py` e `pytest -m integration`, e `npm run docs:check`).
- Abrir o Pull Request de `feat/g7-hybrid` para `develop`.

### 6.7 Roadmap de Commits e Checklist do que Falta (Trilha até a Entrega Final)

Para manter a rastreabilidade científica exigida em `docs/missoes.md` (separando claramente a criação do ferramental, o congelamento do gabarito revisado por humanos e a coleta/avaliação final), o restante do projeto seguirá **4 commits atômicos e auditáveis** na branch `feat/g7-hybrid` (além da manutenção da branch enxuta `g7-hybrid`):

#### Commit A — Sincronização `upstream/develop` (`923cd03`), Unificação Algorítmica e Documentação do Piloto
- [x] Rebase de `feat/g7-hybrid` sobre `upstream/develop` (`923cd03`), integrando `tagging_method` e `TAGGING_METHOD_ENV` em `src/rag_lab/config.py` sem alterar `config/lab.yaml`.
- [x] Unificação de `src/rag_lab/modules/hybrid.py` com `_resolve_hybrid_config` (precedência `settings.hybrid` -> `payload["attributes"]["hybrid_config"]` -> `("rrf", {})`) e teto `0 <= w <= 100` em runtime.
- [x] Auditoria algorítmica do desempate `(-e["score"], e["id"])` e de `boundary_tie`, adicionando 4 novos testes de regressão em `tests/test_hybrid.py` (39 casos em `test_hybrid.py`; `121 passed, 2 skipped` na suíte total).
- [x] Atualização aditiva de `docs/g7-plano-implementacao.md` registrando a auditoria, o piloto ao vivo (`L02` e `S02`) e este roadmap.
- [x] Revisão final do working tree pelo grupo e execução do `git commit` (Commit A `86ae665`) em `feat/g7-hybrid`.

#### Commit B — Fase F9: Ferramental de Coleta, Replay Offline Exato e Testes de Métricas (Pré-Homologação)
- [x] Implementar `experiments/g7/rodar.py` (execução controlada por YAML, contador de chamadas `chat_calls`/`embed_calls`, gravação de envelopes em `data/results/g7/<run_id>/envelopes/` e geração automática de `manifest.json` com hashes SHA-256 e metadados de ambiente).
- [x] Implementar `experiments/g7/avaliar.py` (motor de *replay* offline exato para varredura de `rrf_k`, permutações de *round-robin* e estratégias de fusão; verificação automática `replay_ids == online_ids`; geração das tabelas CSV/Markdown **T1 a T6** e exportação de planilha/CSV para avaliação humana de fidelidade e citações).
- [x] Implementar `tests/test_g7_avaliacao.py` (testes unitários determinísticos para `Recall@k`, `Hit@k`, `MRR`, `nDCG@k`, Jaccard, extração de citações `[n]`, taxa de abstenção e equivalência do motor de *replay* offline).
- [x] Validar `ruff check .` e `pytest` com `test_g7_avaliacao.py` verde (`133 passed, 2 skipped`) e registrar a Fase F9 em `docs/g7-plano-implementacao.md`.

#### Commit C — Conclusão da Fase F8: Homologação Humana do Gabarito e Congelamento do SHA-256
- [ ] Revisão humana cruzada das 24 perguntas e dos 35 trechos relevantes em `experiments/g7/perguntas_gabarito.yaml` por Diego, Gabriel Alves e João Victor Póvoa.
- [ ] Atualização do campo `status_revisao` de `"pendente_validacao_humana"` para `"validado"` (ou `"homologado"`) nos 24 itens.
- [ ] Cálculo e congelamento do hash **SHA-256 definitivo** de `experiments/g7/perguntas_gabarito.yaml` em `experiments/g7/README.md` e `docs/g7-plano-implementacao.md` **antes** de iniciar a coleta oficial da Fase F10.

#### Commit D — Fases F10 e F11: Execução Oficial da Matriz E0–E6, Avaliação Humana e Relatório Final `docs/g7-hybrid.md`
- [ ] Definir e fixar o par oficial de modelos no Ollama para a bateria E0–E6 (padrão dos YAMLs `llama3.2:3b` + `nomic-embed-text` / `embeddinggemma:300m` ou par validado no piloto `lfm3:1.7b-q4_K_M` + `nomic-embed-text:latest`, documentando digests exatos no `manifest.json`).
- [ ] Executar a ingestão única do corpus (`12` arquivos `.md`, `135` chunks) e rodar a matriz experimental completa **E0 a E6** via `experiments/g7/rodar.py` e `experiments/g7/avaliar.py`.
- [ ] Realizar a avaliação humana das respostas geradas (rubrica 0–2 de fidelidade ao contexto, precisão das citações `[n]` e abstenção em `N01`–`N06`) e consolidar as tabelas **T1 a T6** em `experiments/g7/results/`.
- [ ] Redigir o relatório técnico final da missão em `docs/g7-hybrid.md` (pergunta de pesquisa, metodologia, tabelas T1–T6, análise matemática e empírica de *Fusion Tax*, comparação RRF vs. Weighted RRF vs. Convex vs. DBSF vs. Round-Robin, acoplamento com `reranking` e recomendações práticas).
- [ ] Executar a validação final completa fora do sandbox: `ruff check .`, `rag-lab validate`, `rag-lab doctor`, `pytest` completo (incluindo `tests/test_api.py` e `pytest -m integration`) e `npm run docs:check`.
- [ ] Sincronizar se necessário a branch de arquivo único `g7-hybrid` (`src/rag_lab/modules/hybrid.py` já está 100% idêntico entre as duas branches) e abrir o Pull Request final.

---

## 7. Protocolo de Atualização Contínua (Regra de Ouro por Fase)

### 7.1 Regra Operacional Obrigatória

> **REGRA DE OURO:** Ao concluir qualquer fase futura (**F3 até F11**), antes de solicitar aprovação para a fase seguinte, este arquivo (`docs/g7-plano-implementacao.md`) **deve ser atualizado**:
> 1. Marcar `[x] Concluída` e preencher a coluna `Data / Checkpoint` na tabela da **Seção 1.2**;
> 2. Adicionar uma nova entrada completa na **Seção 7.3 (Diário de Atualizações por Fase)** seguindo o template da **Seção 7.2**, documentando os arquivos tocados, os trechos de código reais implementados, as justificativas técnicas e a saída exata dos comandos de verificação.

### 7.2 Template Padrão para Novas Entradas no Diário

```markdown
#### Fase F<N> — <Título da Fase> (<YYYY-MM-DD>)
- **Resumo:** <O que foi entregue nesta fase>
- **Arquivos tocados:** <Lista de arquivos criados/editados com contagem de linhas>
- **Trechos-chave / Mudanças de design:**
  <Blocos de código reais implementados ou decisões tomadas>
- **Por que fizemos assim:** <Justificativa técnica ligada aos contratos e invariantes>
- **Resultados de validação:** <Saída real de ruff, validate, pytest>
- **Pendências / Descobertas:** <Observações para as fases seguintes>
```

### 7.3 Diário de Atualizações por Fase (Changelog)

#### Fase F0 — Setup do Ambiente, Branch `feat/g7-hybrid` e Baseline (`2026-11-18`)
- **Resumo:** Instalação do `uv 0.13.9`, criação da branch `feat/g7-hybrid` a partir de `develop` (`e230073`), provisionamento do `.venv` com Python `3.12.16` via `uv sync --frozen --extra dev --python 3.12` e execução de toda a suíte de verificação sem alterar nenhum arquivo do repositório.
- **Arquivos tocados:** Nenhum arquivo versionado alterado (`working tree clean`).
- **Trechos-chave / Mudanças de design:** Uso de `--frozen` para respeitar integralmente `uv.lock` (66 pacotes instalados).
- **Por que fizemos assim:** Confirmar que a base `develop` estava 100% íntegra antes de introduzir código do G7 e isolar o desenvolvimento na branch da missão.
- **Resultados de validação:**
  - `rag-lab validate`: `{"valid": true}` (exit 0)
  - `rag-lab doctor --offline`: `{"ok": true}` com Python `3.12.16` (exit 0)
  - `ruff check .`: `All checks passed!` (exit 0)
  - `pytest` (suíte unitária sem `test_api.py` no sandbox): `31 passed in 2.20s` (exit 0)
- **Pendências / Descobertas:** Identificado que `tests/test_api.py` utiliza `TestClient` (`anyio` *blocking portal*), cujo `socket.socketpair()` no Windows requer execução fora do sandbox restrito de rede.

#### Fase F1 — Testes de Caracterização em `tests/test_hybrid.py` (`2026-11-18`)
- **Resumo:** Criação da suíte `tests/test_hybrid.py` caracterizando o RRF de referência (`rrf_k = 60`), a mudança de regime com `rrf_k = 1` vs. `60`, o limiar matemático de dominância em $C = 12$ ($k \in \{9, 10, 11\}$), a não-truncagem para `top_k`, a imutabilidade de `payload["retrieval"]`, o tratamento de listas vazias/empates e o viés posicional de `select_without_fusion`.
- **Arquivos tocados:** `tests/test_hybrid.py` (novo, 183 linhas).
- **Trechos-chave / Mudanças de design:**
  - Funções implementadas: `test_rrf_classic_reference_behavior_and_exact_scores_keys`, `test_rrf_k_parameter_shifts_consensus_vs_single_top_winner`, `test_rrf_dominance_threshold_at_candidate_k_12`, `test_hybrid_does_not_truncate_to_top_k_and_does_not_mutate_retrieval`, `test_hybrid_handles_empty_retriever_lists_and_ties_by_id`, `test_select_without_fusion_round_robin_order_bias_and_score_preservation`.
- **Por que fizemos assim:** Estabelecer testes executáveis contra o código original antes de refatorar `hybrid.py`, garantindo zero regressão no comportamento padrão e zero conflito de merge em `tests/test_modules.py`.
- **Resultados de validação:**
  - `ruff check tests/test_hybrid.py`: `All checks passed!`
  - `pytest tests/test_hybrid.py -v`: `9 passed in 1.42s`
- **Pendências / Descobertas:** Cuidado com o desempate lexicográfico por `id` nos candidatos *filler* de testes sintéticos — prefixar *fillers* com `z_` evita empates acidentais com candidatos de interesse no topo.

#### Fase F2 — Contrato YAML `HybridConfig` em `src/rag_lab/config.py` (`2026-11-18`)
- **Resumo:** Extensão retrocompatível de `src/rag_lab/config.py` com `HybridConfig` (`strategy` e `weights`) opcional em `Settings`, validação estrita de tipos e validação cruzada em `Settings.validate_pipeline`, acompanhada de 15 novos casos de teste em `tests/test_config.py`.
- **Arquivos tocados:**
  - `src/rag_lab/config.py` (+40/-5 linhas)
  - `tests/test_config.py` (+55 linhas)
- **Trechos-chave / Mudanças de design:**
  - `Weight = Annotated[StrictFloat | StrictInt, Field(ge=0, le=100, allow_inf_nan=False), AfterValidator(float)]`
  - `HybridConfig(StrictModel)` com `strategy: FusionStrategy = "rrf"` e `weights: dict[RetrieverName, Weight] = Field(default_factory=dict)`
  - Validação cruzada em `Settings.validate_pipeline` ativa apenas quando `self.modules["hybrid"].enabled` é `True`.
- **Por que fizemos assim:** Manter `config/lab.yaml` intocado e válido por padrão, impedir coerções silenciosas de `bool`/`str` ou valores `NaN`/`Inf` nos pesos, e bloquear combinações ambíguas no YAML (`rrf` com `weights`, `weighted_rrf` sem `weights`, chaves de `weights` divergentes dos recuperadores ativos ou soma de pesos $\le 0$).
- **Resultados de validação:**
  - `ruff check .`: `All checks passed!`
  - `rag-lab --config config/lab.yaml validate`: `{"valid": true}`
  - `pytest --ignore=tests/test_api.py -v`: `55 passed, 1 skipped in 2.05s`
- **Pendências / Descobertas:** Na F3, `Hybrid.run` também validará `weights` contra `payload["retrieval"].keys()` em tempo de execução para cobrir chamadas diretas a `Hybrid().run(payload, services)` em testes unitários que não passam por `Settings.validate_pipeline`.

#### Fases F3 a F6 — Núcleo Strategy (`rrf`, `weighted_rrf`, `convex`, `dbsf`), `math.fsum` e Telemetria (`2026-11-18`)
- **Resumo:** Evolução completa de `src/rag_lab/modules/hybrid.py` com o padrão *Strategy* (`STRATEGIES`), agregação exata via `math.fsum`, validação em runtime de scores brutos e pesos contra as chaves de `payload["retrieval"]`, telemetria diagnóstica em `payload["attributes"]["hybrid"]` (`schema: "g7-hybrid/1"`) e expansão de `tests/test_hybrid.py` para 15 funções de teste (32 casos de teste parametrizados).
- **Arquivos tocados:**
  - `src/rag_lab/modules/hybrid.py` (reescrito, 222 linhas; `+223/-12` no diff)
  - `tests/test_hybrid.py` (494 linhas totais; 32 casos de teste parametrizados)
- **Trechos-chave / Mudanças de design:**
  ```python
  def _score_convex(
      raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
  ) -> list[float]:
      if not raw_scores or weight_sum <= 0.0:
          return []
      factor = weight / weight_sum
      lo = min(raw_scores)
      hi = max(raw_scores)
      span = hi - lo
      if span <= 0.0:
          return [factor for _ in raw_scores]
      return [factor * ((score - lo) / span) for score in raw_scores]


  def _score_dbsf(
      raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
  ) -> list[float]:
      if not raw_scores or weight_sum <= 0.0:
          return []
      factor = weight / weight_sum
      mean = fmean(raw_scores)
      std = pstdev(raw_scores)
      if std <= 0.0:
          return [factor for _ in raw_scores]
      lo = mean - 3.0 * std
      span = 6.0 * std
      return [factor * min(1.0, max(0.0, (score - lo) / span)) for score in raw_scores]
  ```
- **Por que fizemos assim:**
  1. **Derivação de `retrievers = list(retrieval.keys())`:** garante que testes unitários que invocam `Hybrid().run(payload, services)` diretamente sem alterar os switches de `services.settings.modules` funcionem sem acoplamento artificial, ao mesmo tempo em que no `Pipeline.run` (onde cada recuperador ativo chama `add_ranking`) o denominador `weight_sum = math.fsum(weights.values())` inclui todos os recuperadores ativos, mesmo os que retornam `[]`.
  2. **Pré-divisão das contribuições por `weight_sum`:** faz com que `math.fsum(entry["contributions"].values()) == entry["fused"]` feche bit-a-bit em todas as 4 estratégias.
  3. **`math.fsum`:** elimina a não-associatividade da soma float padrão (demonstrada em teste com a tupla de ranks `(1, 7, 2)` vs. `(2, 7, 1)` em `rrf_k = 60`, onde `((1/61 + 1/67) + 1/62) < ((1/62 + 1/67) + 1/61)`).
  4. **Preservação estrita de `item["scores"]`:** nenhuma chave extra além de `"hybrid"` é inserida em `item["scores"]`, mantendo `tests/test_modules.py::test_rrf_rewards_agreement_without_adding_raw_scores` intacto e verde.
- **Resultados de validação:**
  - `ruff check src/rag_lab/modules/hybrid.py tests/test_hybrid.py`: `All checks passed!`
  - `pytest tests/test_hybrid.py tests/test_modules.py -v`: `37 passed in 1.49s`
- **Pendências / Descobertas:** Demonstrado matematicamente que para `rrf_k = 60`, as posições `{1, 2, 7}` produzem discrepância de 1 ULP na adição float padrão (`(1, 7, 2)` vs. `(2, 7, 1)`), corrigida por `math.fsum`.

#### Fase F7 — Testes de Composição Ponta-a-Ponta em `tests/test_pipeline.py` (`2026-11-18`)
- **Resumo:** Adição dos testes de integração de pipeline `test_pipeline_hybrid_strategies_emit_telemetry_and_preserve_contract` (parametrizado para `rrf`, `weighted_rrf`, `convex` e `dbsf`) e `test_pipeline_without_hybrid_uses_round_robin_and_leaves_hybrid_bypassed` em `tests/test_pipeline.py`.
- **Arquivos tocados:** `tests/test_pipeline.py` (+45 linhas).
- **Trechos-chave / Mudanças de design:**
  - Verificação de que `result.attributes["hybrid"]["ranking"]` preserva a lista fundida completa (`len(ranking) >= len(result.evidence)`) após `build_context` truncar `result.evidence` para `top_k = 2`.
  - Verificação de que com `hybrid` desativado o `trace` registra `status == "bypass"` e nem `result.attributes` nem `result.evidence[0].scores` recebem a chave `"hybrid"`.
- **Por que fizemos assim:** Validar que o round-trip JSON (`json.loads(json.dumps(payload))`) e o `Envelope.model_validate` executados pelo `Pipeline.run` após o módulo `hybrid` aceitam todas as 4 estratégias e sua telemetria sem perda de dados.
- **Resultados de validação:**
  - `ruff check .`: `All checks passed!`
  - `rag-lab --config config/lab.yaml validate`: `{"valid": true}`
  - `pytest --ignore=tests/test_api.py -v`: `83 passed, 1 skipped in 2.03s`
- **Pendências / Descobertas:** Código de produção do módulo `hybrid.py` e `config.py` concluído e 100% coberto por testes; pronto para iniciar a Fase F8 (construção do corpus, perguntas e gabarito experimental em `experiments/g7/`).

#### Fase F8 — Protocolo Experimental, Corpus Autoral, Gabarito e Configs YAML (`2026-11-18`)
- **Resumo:** Construção completa do pacote experimental independente em `experiments/g7/`: 12 documentos `.md` autorais (licença CC-BY-4.0) sobre a Plataforma Fictícia "Núcleo Aurora" (`23.991` caracteres, `135` chunks sentenciais), 24 perguntas balanceadas em 4 perfis (`6` literais, `6` semânticas, `6` compostas, `6` sem resposta) com gabarito de dupla anotação em `experiments/g7/perguntas_gabarito.yaml` (`35` trechos relevantes), 13 configurações YAML em `experiments/g7/configs/` e documentação completa em `experiments/g7/README.md`.
- **Arquivos tocados:**
  - `experiments/g7/corpus/01_arquitetura_geral.md` .. `12_governanca_ingestao_dados.md` (12 novos arquivos `.md`, `23.991` caracteres totais, `135` chunks)
  - `experiments/g7/perguntas_gabarito.yaml` (novo, 492 linhas, 24 perguntas, 35 trechos relevantes)
  - `experiments/g7/configs/*.yaml` (13 novos arquivos YAML cobrindo E0, E1, E2, E4 e E5)
  - `experiments/g7/README.md` (novo, 73 linhas)
- **Trechos-chave / Mudanças de design:**
  - Cada sentença do corpus foi redigida sem abreviações internas com ponto seguido de espaço e com comprimento inferior a `1000` caracteres, garantindo alinhamento determinístico 1:1 com a regex `re.split(r"(?<=[.!?])\s+|\n+", text)` de `src/rag_lab/ingestion.py`.
  - Inclusão deliberada dos 5 mecanismos de estresse: (1) códigos exatos (`ERR-AUR-417`, `POL-SEG-09`, `SEC-LCK-301`, `ALR-LAT-901`); (2) paráfrases semânticas sem sobreposição lexical (`S01`–`S06`); (3) distratores lexicais entre domínios ("token" em autenticação vs. faturamento; "rotação" em chaves Ed25519 vs. logs de 500 MB; "expurgo"/"retenção" em DLQ de 21 dias vs. backup WORM de 90 dias vs. auditoria de 365 dias/5 anos; "janela" em 5 documentos); (4) pares quase redundantes (`03`+`07`, `09`+`10`, `06`+`11`); e (5) fatos complementares multi-documento (`C01`–`C06`).
  - Schema do gabarito com casamento por substring normalizada contida em chunk único e governança de revisão cruzada (`anotador`, `revisor`, `status_revisao: pendente_validacao_humana`).
- **Por que fizemos assim:**
  - Bloquear o uso de conhecimento prévio do LLM (`llama3.2:3b`) por meio de um domínio fictício controlado, cumprir rigorosamente o protocolo científico de `docs/missoes.md` (gabarito independente criado antes de rodar o sistema, base autorizada sob CC-BY-4.0, controle de variáveis mantendo `top_k: 5`, `candidate_k: 12` e `max_context_chars: 12000` constantes nos 13 YAMLs) e permitir versionamento no Git fora das pastas ignoradas `data/documents/` e `data/results/`.
- **Resultados de validação:**
  - `rag-lab --config <arquivo> validate` nos 13 YAMLs de `experiments/g7/configs/`: `13/13 EXIT=0`.
  - Script de verificação de corpus e gabarito: `12/12` arquivos entre `1.826` e `2.234` chars (`135` chunks), `24/24` perguntas (`6` por perfil) e `35/35` `trechos_relevantes` casando exatamente no arquivo e dentro de um único chunk sentencial de `split_document`.
  - SHA-256 inicial pré-revisão humana de `experiments/g7/perguntas_gabarito.yaml`: `046841378bc39d6dd95fd2edefe37f3d91161afed85802925d6e932c68762e32`.
- **Pendências / Descobertas:** Revisão humana cruzada pelo trio (Diego, Gabriel Alves e João Victor Póvoa) para alterar `status_revisao` de `"pendente_validacao_humana"` para `"validado"`/`"homologado"` e congelar o SHA-256 definitivo antes da coleta oficial na F10; pronto para iniciar a Fase F9 (`rodar.py`, `avaliar.py` e `tests/test_g7_avaliacao.py`).

#### Sincronização com `upstream/develop` (`923cd03`), Auditoria Algorítmica e Piloto Ao Vivo (`2026-11-26`)
- **Resumo:** Rebase completo de `feat/g7-hybrid` sobre `upstream/develop` (`923cd03`), unificação de `src/rag_lab/modules/hybrid.py` com o fallback autocontido `_resolve_hybrid_config` da branch `g7-hybrid`, auditoria matemática linha a linha do critério de desempate `(-e["score"], e["id"])` e de `boundary_tie`, adição de 4 novos testes de regressão em `tests/test_hybrid.py` (elevando a suíte unitária para `121 passed, 2 skipped`) e registro completo do piloto ao vivo (`L02` e `S02` nos 5 modos em Docker Compose + Ollama).
- **Arquivos tocados:**
  - `src/rag_lab/config.py` (merge limpo preservando `TaggingMethod` / `TAGGING_METHOD_ENV` de `upstream/develop` e `HybridConfig` / `Weight` do G7)
  - `src/rag_lab/modules/hybrid.py` (267 linhas; unificado com `_resolve_hybrid_config` e validação `0 <= w <= 100` em runtime)
  - `tests/test_config.py` (ajuste de `config_data["tagging_method"] = "llm"` nos 2 testes diretos de `Settings.model_validate` do G7)
  - `tests/test_hybrid.py` (672 linhas; +180 linhas com 4 novos testes de regressão, totalizando 19 funções / 39 casos parametrizados)
  - `docs/g7-plano-implementacao.md` (atualização aditiva do painel de progresso ASCII, Seção 5.7, Seção 6.7 e este registro no changelog)
- **Trechos-chave / Mudanças de design:**
  - Unificação de `_resolve_hybrid_config(payload, settings)` com prioridade `settings.hybrid` -> `payload["attributes"]["hybrid_config"]` -> `("rrf", {})`, tornando `src/rag_lab/modules/hybrid.py` 100% idêntico entre `feat/g7-hybrid` e `g7-hybrid`.
  - Prova formal e empírica de que `ordered = sorted(merged.values(), key=lambda e: (-e["score"], e["id"]))` é estritamente invariante à escala bruta dos recuperadores: em `S02__rrf60.json`, o empate em $r=3$ (`1/63`) foi decidido lexicograficamente pelos hashes SHA-256 (`"54833391d69b..." < "e59f8e132b52..."`), enquanto nos empates em $r=5$ (`1/65`) e $r=6$ (`1/66`) os candidatos do `embedding` (`raw_score ~ 0.67`) venceram os do `bm25` (`raw_score ~ 8.5–8.9`) porque `"909b83e155a1..." < "df6be78ee17c..."` e `"107492924e6b..." < "424a91ce6ffb..."`.
- **Por que fizemos assim:**
  - Manter compatibilidade simultânea com a entrega de arquivo único (`g7-hybrid` sobre `upstream/develop` sem tocar em `config.py`) e com a arquitetura completa do projeto (`feat/g7-hybrid`), blindando com testes de regressão a invariância de escala no desempate, a preservação de `metadata["tags"]` e a detecção de `boundary_tie` na fronteira `top_k`.
- **Resultados de validação:**
  - `uv run --no-sync python -m ruff check .`: `All checks passed!`
  - `uv run --no-sync python -m rag_lab.cli validate` (`config/lab.yaml` + 13 YAMLs de `experiments/g7/configs/*.yaml`): `14/14 {"valid": true}`
  - `uv run --no-sync python -m pytest --ignore=tests/test_api.py -v`: `121 passed, 2 skipped in 3.12s`
  - Reavaliação offline dos envelopes do piloto ao vivo (`L02__rrf60.json` e `S02__rrf60.json`): `identical_ranking=True` em ambos (`target_fused_rank=1` em `L02`, `target_fused_rank=7` em `S02`, `boundary_tie=False`).
- **Pendências / Descobertas:** Working tree pronta para o **Commit A** em `feat/g7-hybrid`; próximo passo imediato após o commit é iniciar a Fase F9 (**Commit B**: `experiments/g7/rodar.py`, `experiments/g7/avaliar.py` e `tests/test_g7_avaliacao.py`).

#### Fase F9 — Ferramental de Coleta, Replay Offline Exato, Métricas e Testes (`2026-11-26`)
- **Resumo:** Implementação completa do coletor experimental controlado (`experiments/g7/rodar.py`), do avaliador de métricas com motor de *replay* offline exato e gerador das tabelas T1–T6 e template humano (`experiments/g7/avaliar.py`), e da suíte de testes unitários e de integração em memória (`tests/test_g7_avaliacao.py`).
- **Arquivos tocados:**
  - `experiments/g7/rodar.py` (novo, 385 linhas)
  - `experiments/g7/avaliar.py` (novo, 743 linhas)
  - `tests/test_g7_avaliacao.py` (novo, 335 linhas, 7 funções / 12 casos parametrizados)
  - `docs/g7-plano-implementacao.md` (atualização aditiva do painel de progresso ASCII, Seção 5.8, checklist do Commit B e este registro no changelog)
- **Trechos-chave / Mudanças de design:**
  - `InstrumentedProviders` em `rodar.py` para capturar `chat_calls`, `embed_calls` e `embed_texts` por consulta e `sanitize_settings` para garantir zero vazamento de `database_url` ou senhas em `manifest.json`.
  - `compute_document_metrics` e `compute_snippet_metrics` em `avaliar.py` com reconstrução do texto completo via `build_text_lookup(envelope)` sobre `attributes.hybrid.ranking`, permitindo calcular `Recall@10` exato mesmo após o corte `top_k = 5` de `build_context`.
  - `replay_from_retrieval` invocando diretamente `select_without_fusion` e `Hybrid().run` para varrer `rrf_k` (**E2**), permutações de ordem do *round-robin* (**E3**) e estratégias de fusão (**E4**), com verificação automática `verify_replay_equivalence`.
- **Por que fizemos assim:**
  - Garantir que todas as comparações de `rrf_k`, permutações do *round-robin* e estratégias de fusão operem sobre exatamente os mesmos candidatos recuperados, eliminando variância de rede/banco e validando matematicamente `replay == online`.
- **Resultados de validação:**
  - `uv run --no-sync python -m ruff check .`: `All checks passed!`
  - `uv run --no-sync python -m pytest --ignore=tests/test_api.py -v`: `133 passed, 2 skipped in 5.03s` (12 novos casos em `tests/test_g7_avaliacao.py`)
  - `uv run --no-sync python experiments/g7/rodar.py --config experiments/g7/configs/e2_rrf_k60_bm25_emb.yaml --dry-run`: manifesto `g7-run-manifest/1` gerado com `selected_questions: 24`, hashes SHA-256 dos 12 arquivos do corpus e zero segredos.
  - `uv run --no-sync python experiments/g7/avaliar.py --run-dir data/results/teste_ao_vivo --out-dir data/results/teste_ao_vivo/avaliacao_piloto`: `10/10` envelopes avaliados, `replay_verification.all_matched = true`, 4 casos de *Fusion Tax* em `S02` reportados em `t5_fusion_tax.{csv,md}` e `avaliacao_humana_template.csv` gerado.
- **Pendências / Descobertas:** Ferramental da Fase F9 pronto para o **Commit B**; próxima etapa é a revisão humana final e congelamento do SHA-256 de `experiments/g7/perguntas_gabarito.yaml` (**Commit C**) seguida da execução oficial da matriz E0–E6 (**Fase F10 / Commit D**).
