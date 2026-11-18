# Protocolo Experimental G7 — Fusão Híbrida (`hybrid.py`)

## 1. Autoria, Licença e Conformidade Científica

- **Grupo:** G7 — Hybrid (`src/rag_lab/modules/hybrid.py`)
- **Autores:** Diego · Gabriel Alves · João Victor Póvoa
- **Licença do Corpus e do Gabarito:** Creative Commons Attribution 4.0 International (**CC-BY-4.0**). Todo o acervo em `experiments/g7/corpus/` é material didático autoral e fictício elaborado especificamente pelo Grupo G7 para avaliação controlada de Recuperação de Informação no `rag-lab`, livre de restrições de direitos autorais de terceiros e de dados pessoais.
- **Independência do Gabarito (`docs/missoes.md`):** O arquivo `experiments/g7/perguntas_gabarito.yaml` foi redigido e anotado manualmente a partir dos 12 documentos do corpus **antes** de qualquer execução do pipeline ou do modelo de linguagem. Em nenhuma hipótese respostas do próprio sistema são utilizadas como gabarito.

---

## 2. Mapa dos 12 Documentos do Corpus ("Plataforma Núcleo Aurora")

Cada documento foi redigido em português (UTF-8), com **1.500 a 2.800 caracteres** e sentenças autocontidas compatíveis com o segmentador `split_document(..., max_chars=1000)` de `src/rag_lab/ingestion.py`:

| Arquivo | Título / Domínio | Papel no Desenho Experimental |
|---|---|---|
| `01_arquitetura_geral.md` | Arquitetura Geral e Topologia | Portas (`9443`, `5353`), limite `380` conexões, coordenador `aurora-core`; fato complementar para `S04` e `C06` |
| `02_autenticacao_identidade.md` | Autenticação, Identidade e Acesso | Protocolo `AUR-ID-702`, norma `POL-SEG-09` (`18` min), evento `SEC-LCK-301`, rotação Ed25519 (`14` dias); distrator lexical de "token" e "rotação" (`L01`, `C01`) |
| `03_banco_vetorial_indexacao.md` | Banco Vetorial e Indexação Semântica | `aurora-vectordb`, `768` dimensões, erro `ERR-AUR-417`, busca exata até `50000` trechos, rollback transacional; par quase redundante com `07` (`L02`, `S01`, `C04`, `N02`) |
| `04_filas_mensageria.md` | Filas de Mensageria Assíncrona | `aurora-queue`, janela `120` s, erro `ERR-AUR-508`, fila `aurora.dlq.critico`, expurgo em `21` dias; distrator lexical de "janela" e "expurgo" (`L03`) |
| `05_faturamento_cotas.md` | Faturamento, Cotas e Bilhetagem | `aurora-billing`, franquia `4500000` tokens, erro `ERR-AUR-429`, isenção sem síntese, custo do preflight; distrator lexical de "token" (`L04`, `S04`, `C05`, `N03`) |
| `06_seguranca_criptografia.md` | Segurança, Rede e Criptografia | `POL-SEG-14` (`aurora_db_net`, porta `5432`), `aurora-vault` (recifragem semestral), sanitização HTTP `503`; alvo de `L05`, `S02` e `C01` |
| `07_backup_recuperacao.md` | Backup e Recuperação de Desastres | `aurora-backup`, retenção WORM `90` dias, RPO `15` min, RTO `40` min, transação atômica por documento; par com `03`, `08` e `10` (`S01`, `C02`, `C03`) |
| `08_auditoria_conformidade.md` | Auditoria e Conformidade Legal | `POL-AUD-03`, expurgo em `365` dias, retenção fiscal `5` anos, proibição de gabarito automático; alvo de `S06` e `C02` |
| `09_observabilidade_telemetria.md` | Observabilidade e Telemetria | OpenTelemetry porta `4317`, rotação de log `500` MB, alerta `ALR-LAT-901` (`4500` ms), rota `/health` sem LLM; alvo de `L06`, `C05` e `C06` |
| `10_slas_disponibilidade.md` | SLAs e Disponibilidade | `POL-SLA-01`, `99,95%` (`21` min/mês), crédito de `12%`, janela de manutenção terças `02h00–04h00`; alvo de `C03` e distrator de `N03` |
| `11_deploy_configuracao.md` | Implantação e Configuração Estrita | `AURORA_CONFIG_PATH`, proibição de recarga dinâmica (exige reinício), booleanos estritos; alvo de `S03` |
| `12_governanca_ingestao_dados.md` | Governança e Pipeline de Ingestão | `aurora-ingest`, `.txt`/`.md` UTF-8, identidade por caminho relativo, obrigatoriedade de reingestão ao mudar modelo/dimensão; alvo de `S05`, `C04` e `N01` |

---

## 3. Estrutura das 24 Perguntas (`perguntas_gabarito.yaml`)

- **`L01`–`L06` (`literal`):** Buscam siglas, códigos de erro (`ERR-AUR-417`, `ERR-AUR-429`, `SEC-LCK-301`, `ALR-LAT-901`), políticas (`POL-SEG-09`, `POL-SEG-14`) e portas/quantidades exatas.
- **`S01`–`S06` (`semantica`):** Paráfrases naturais que evitam os termos técnicos raros das sentenças-alvo, exigindo representação vetorial densa e testando se o viés lexical 2:1 do RRF prejudica o recuperador `embedding`.
- **`C01`–`C06` (`composta`):** Exigem recuperar e sintetizar fatos distribuídos em dois documentos distintos dentro do orçamento `top_k: 5`.
- **`N01`–`N06` (`sem_resposta`):**
  - `N01`–`N03` (*near-miss*): citam subsistemas reais (`aurora-ingest`, `aurora-vectordb`, `aurora-billing`), mas perguntam sobre parâmetros inexistentes ou funcionalidades explicitamente vedadas no corpus.
  - `N04`–`N06` (*fora do domínio*): temas completamente externos (Copa de 2002, culinária, biologia vegetal).

---

## 4. Matriz de Configurações YAML (`experiments/g7/configs/`)

Todas as configurações mantêm constantes: `top_k: 5`, `candidate_k: 12`, `max_context_chars: 12000`, `max_sentence_chars: 1000`, `llm.temperature: 0.0` e `database_url: ${DATABASE_URL:-postgresql://rag:rag@localhost:5432/rag_g7}`.

| Arquivo YAML | Bloco | Recuperadores Ativos | `hybrid` | Estratégia / Parâmetros | `reranking` |
|---|---|---|---|---|---|
| `e0_tfidf.yaml` | E0 (Baseline) | `tfidf` | `false` | — | `false` |
| `e0_bm25.yaml` | E0 (Baseline) | `bm25` | `false` | — | `false` |
| `e0_embedding.yaml` | E0 (Baseline) | `embedding` | `false` | — | `false` |
| `e1_rr_bm25_emb.yaml` | E1 (Round-Robin) | `bm25`, `embedding` | `false` | `select_without_fusion` | `false` |
| `e1_rr_todos.yaml` | E1 (Round-Robin + Mestre Replay) | `tfidf`, `bm25`, `embedding` | `false` | `select_without_fusion` | `false` |
| `e2_rrf_k10_bm25_emb.yaml` | E2 (Limiar `rrf_k=10`) | `bm25`, `embedding` | `true` | `rrf` (`rrf_k: 10`) | `false` |
| `e2_rrf_k60_bm25_emb.yaml` | E2 (Padrão `rrf_k=60`) | `bm25`, `embedding` | `true` | `rrf` (`rrf_k: 60`) | `false` |
| `e2_rrf_k60_todos.yaml` | E2 (Padrão 3 recuperadores) | `tfidf`, `bm25`, `embedding` | `true` | `rrf` (`rrf_k: 60`) | `false` |
| `e4_weighted_rrf_todos.yaml` | E4 (Estratégias Avançadas) | `tfidf`, `bm25`, `embedding` | `true` | `weighted_rrf` (`1.0, 1.0, 2.0`) | `false` |
| `e4_convex_bm25_emb.yaml` | E4 (Estratégias Avançadas) | `bm25`, `embedding` | `true` | `convex` (`1.0, 1.0`) | `false` |
| `e4_dbsf_bm25_emb.yaml` | E4 (Estratégias Avançadas) | `bm25`, `embedding` | `true` | `dbsf` (`1.0, 1.0`) | `false` |
| `e5_coupling_rr_rerank.yaml` | E5 (Acoplamento Reranking) | `bm25`, `embedding` | `false` | `select_without_fusion` | `true` |
| `e5_coupling_rrf_rerank.yaml` | E5 (Acoplamento Reranking) | `bm25`, `embedding` | `true` | `rrf` (`rrf_k: 60`) | `true` |

---

## 5. Protocolo de Revisão Humana e Congelamento do Gabarito

1. **Revisão Cruzada:** Cada pergunta em `perguntas_gabarito.yaml` possui um `anotador` e um `revisor` designados entre Diego, Gabriel Alves e João Victor Póvoa.
2. **Critério de Homologação:** Verificar se os `trechos_relevantes` são necessários e suficientes para sustentar os `fatos_esperados` e se nenhuma pergunta `sem_resposta` possui resposta acidental no corpus.
3. **Congelamento do Hash SHA-256:** Após a revisão humana, atualizar `status_revisao` para `"homologado"` e registrar o hash SHA-256 de `perguntas_gabarito.yaml` no manifesto das execuções da Fase F10:
   ```powershell
   .\.venv\Scripts\python.exe -c "import hashlib, pathlib; p = pathlib.Path('experiments/g7/perguntas_gabarito.yaml'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
   ```
