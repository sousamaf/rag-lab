# G7 · Hybrid — Documento Vivo de Engenharia e Planejamento

> **Documento vivo da missão G7 (`hybrid.py`).** Este arquivo registra a arquitetura, as decisões de engenharia (ADRs), a fundamentação matemática, o código já implementado com suas justificativas e o plano executável das próximas fases. Deve ser atualizado ao término de cada fase (**F0 a F11**).

---

## 1. Cabeçalho e Painel de Status Vivo

### 1.1 Metadados do Trabalho

| Campo | Valor |
|---|---|
| **Grupo** | **G7 — Hybrid** |
| **Integrantes** | Diego · Gabriel Alves · João Victor Póvoa |
| **Repositório / Branch** | `rag-lab` · branch `feat/g7-hybrid` (criada a partir de `develop`, commit base `e230073`) |
| **Módulo principal** | `src/rag_lab/modules/hybrid.py` (com contrato YAML estendido em `src/rag_lab/config.py`) |
| **Missão oficial (`docs/missoes.md`)** | Comparar seleção básica *round-robin* (`select_without_fusion`) e **Reciprocal Rank Fusion (RRF)** variando `rrf_k`, entregando código sólido, comparação controlada e análise honesta das limitações da técnica |
| **Ambição técnica do G7** | Evoluir `hybrid.py` com arquitetura *Strategy* retrocompatível via YAML (`rrf`, `weighted_rrf`, `convex`, `dbsf`), precisão numérica (`math.fsum`), proteção *NaN-free*, telemetria diagnóstica em `payload["attributes"]["hybrid"]` e suíte de avaliação experimental com *replay* offline exato |
| **Diagramas de referência** | [Fluxo didático](diagrams/rag-didatico.svg) · [Mapa dos módulos e grupos](diagrams/rag-modulos.svg) · [Pipeline modular](diagrams/rag-pipeline.svg) |

### 1.2 Painel de Progresso das Fases (F0 a F11)

| Fase | Status | Arquivos-alvo | Critério de pronto | Data / Checkpoint |
|---|---|---|---|---|
| **F0** | `[x] Concluída` | Ambiente `.venv` (`uv 0.13.9`, CPython `3.12.16`), branch `feat/g7-hybrid` | `uv sync --frozen --extra dev --python 3.12`, `rag-lab validate`, `rag-lab doctor --offline`, `ruff check .` e `pytest` verdes sem editar arquivos | 2026-11-18 · Base `e230073` validada (31/31 testes unitários verdes no sandbox) |
| **F1** | `[x] Concluída` | `tests/test_hybrid.py` (novo) | Testes de caracterização do RRF clássico, variação de `rrf_k`, limiar de dominância $C=12$, não-truncagem, imutabilidade de `retrieval` e viés do *round-robin* passando contra o código original | 2026-11-18 · `9 passed` contra `hybrid.py` original + `ruff` limpo |
| **F2** | `[x] Concluída` | `src/rag_lab/config.py`, `tests/test_config.py` | `HybridConfig` opcional na raiz do YAML mantendo `config/lab.yaml` intocado; validação estrita de `strategy` e `weights`; testes de contrato verdes | 2026-11-18 · `55 passed, 1 skipped` + `validate` e `ruff` limpos |
| **F3** | `[x] Concluída` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Núcleo *Strategy* + `rrf` com `math.fsum` + telemetria `attributes.hybrid` + validação *NaN-free*; teste legado em `tests/test_modules.py` intacto e verde | 2026-11-18 · `STRATEGIES`, `_score_rrf`, `_resolve_weights`, `_extract_raw_scores` e telemetria `g7-hybrid/1` verificados |
| **F4** | `[x] Concluída` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `weighted_rrf` implementada; equivalência com pesos unitários, inversão de vencedor por peso e peso zero sem truncagem testados | 2026-11-18 · `_score_weighted_rrf` e validação em runtime contra `payload["retrieval"]` verificados |
| **F5** | `[x] Concluída` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `convex` (Min-Max normalizado por $\sum w_j$) implementada; invariância a escala, sensibilidade a *outlier* e casos degenerados ($\max = \min$, 1 item) testados | 2026-11-18 · `_score_convex` com denominador $W$ sobre todas as chaves de `retrieval` verificado |
| **F6** | `[x] Concluída` | `src/rag_lab/modules/hybrid.py`, `tests/test_hybrid.py` | Estratégia `dbsf` ($\mu \pm 3\sigma$ com *clip* $[0, 1]$ normalizado por $\sum w_j$) implementada; $\sigma = 0$, *clip* em $n=12$ e soma exata `fsum(contributions) == fused` testados | 2026-11-18 · `_score_dbsf` e round-trip JSON estrito nas 4 estratégias verificados (`37 passed` em `test_hybrid`+`test_modules`) |
| **F7** | `[x] Concluída` | `tests/test_pipeline.py` | Testes de composição ponta-a-ponta no `Pipeline` para todas as estratégias, bypass do `hybrid` e preservação do envelope JSON | 2026-11-18 · `83 passed, 1 skipped in 2.03s` + `ruff` e `validate` limpos |
| **F8** | `[ ] Pendente` | `experiments/g7/{corpus/,gabarito.yaml,configs/*.yaml,README.md}` | Corpus autorizado (10–15 `.md`), 24 perguntas (4 perfis) e gabarito com dupla anotação e hash SHA-256 congelado antes das execuções | Aguardando início |
| **F9** | `[ ] Pendente` | `experiments/g7/{rodar.py,avaliar.py}`, `tests/test_g7_avaliacao.py` | Coletor online + avaliador com *replay* offline exato, tabelas T1–T6 e testes unitários das métricas; equivalência replay == online verificada em piloto | Aguardando F8 |
| **F10** | `[ ] Pendente` | `compose.override.yaml` (local, não versionado), `data/results/g7/` | Banco dedicado `rag_g7` via Docker (`127.0.0.1:5432`), ingestão única com embeddings e execução da matriz E0–E6 com manifestos completos | Aguardando F9 |
| **F11** | `[ ] Pendente` | `docs/g7-hybrid.md`, `experiments/g7/results/` | Relatório técnico final com tabelas, casos de *Fusion Tax*, limitações e recomendação; suíte completa (incluindo `test_api.py` e integração) verde; PR `feat/g7-hybrid` -> `develop` | Aguardando F10 |

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

## 5. Registro Detalhado do que Já Foi Feito (F0 a F7)

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
