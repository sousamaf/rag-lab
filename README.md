# Laboratório RAG didático

Base para experimentar **Retrieval-Augmented Generation** em grupos: preparar documentos,
buscar evidências e gerar respostas com indicação das fontes. O fluxo acompanha
[rag-didatico.mmd](docs/rag-didatico.mmd) e as dez missões de
[rag-modulos.mmd](docs/rag-modulos.mmd).

![Fluxo didático de preparação e consulta RAG, renderizado com Pele.js](docs/diagrams/rag-didatico.svg)

Os diagramas são publicados como SVG gerado com **Pele.js**. O GitHub exibe a imagem
pronta; as fontes Mermaid `.mmd` continuam editáveis. Veja a
[galeria e as instruções de geração](docs/diagramas.md).

O código é Python, com PostgreSQL + pgvector e integrações LangChain para chat,
embeddings e divisão de textos. A interface inicial é uma CLI e uma API HTTP com
documentação interativa em [localhost:8000/docs](http://localhost:8000/docs).

## Início com Docker — MacOS e Linux

Requisitos: Git, Docker com Compose v2 e um endpoint de modelos disponível.
Instale e inicie o [Ollama](https://docs.ollama.com/) no computador e baixe os modelos
definidos no YAML:

```sh
ollama pull llama3.2:3b
ollama pull nomic-embed-text
cp .env.example .env
docker compose up --build -d --wait
docker compose exec app rag-lab doctor
docker compose exec app rag-lab ingest examples/documents
docker compose exec app rag-lab query "Como um sistema RAG usa os documentos?"
```

No MacOS, o Ollama roda no host para aproveitar a aceleração disponível. O Compose
acessa o host por `host.docker.internal`; no Linux, o alias é criado por `host-gateway`.
**No Linux, se o Ollama escutar somente em `127.0.0.1`, configure o serviço com
`OLLAMA_HOST=0.0.0.0:11434` e reinicie-o**, limitando o acesso de rede ao ambiente da
aula. Outra opção é definir `OLLAMA_BASE_URL` com um endpoint acessível ao container.

Coloque materiais próprios `.txt` ou `.md` em UTF-8 em `data/documents/` e execute:

```sh
docker compose exec app rag-lab ingest data/documents
```

Os documentos pessoais e as preferências dos assistentes não são versionados. O banco
persiste no volume `postgres_data`. `docker compose down` para os serviços e preserva
os dados; `docker compose logs app` mostra erros de inicialização. Para mudanças no
código, rode novamente `docker compose up --build -d`.

## Início nativo — Windows / PowerShell

1. Instale Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
   PostgreSQL **16 ou superior** e a extensão
   [pgvector](https://github.com/pgvector/pgvector#windows) compatível com essa instalação.
2. Crie um usuário e banco para o laboratório. No banco escolhido, habilite
   `CREATE EXTENSION IF NOT EXISTS vector;` com um usuário autorizado. O usuário de
   execução deve poder criar e acessar tabelas no schema `public`.
3. Instale/inicie Ollama e baixe os dois modelos acima, ou configure endpoints
   compatíveis com OpenAI. Copie `.env.example` para `.env` e ajuste `DATABASE_URL`
   para o banco nativo, incluindo credenciais e porta.
4. Na raiz do repositório, execute:

```powershell
Copy-Item .env.example .env
# Edite .env antes de continuar; não sobrescreva um .env que já esteja configurado.
.\scripts\start.ps1 -Install
```

`-Install` prepara `.venv` com Python 3.12 e as versões de `uv.lock`. Nas próximas
inicializações, use apenas `.\scripts\start.ps1`. A instalação nativa não usa Docker.
Se a política local impedir scripts, siga a política de execução definida pela
instituição para permitir o script neste terminal.

O script verifica o checkout, Python, dependências, YAML, porta da API, conexão e
versão do PostgreSQL, extensão pgvector, permissão de criação, credencial/modelo de
chat e, quando ativo, endpoint e dimensão dos embeddings. Falhas impedem a abertura
da API. O teste de chat faz uma chamada curta ao modelo e pode consumir a cota do
endpoint. `-CheckOnly` executa as verificações e inicializa as tabelas sem abrir a API.

```powershell
.\scripts\start.ps1 -Config config/lab.local.yaml -CheckOnly
.\.venv\Scripts\rag-lab.exe ingest examples/documents
.\.venv\Scripts\rag-lab.exe query "O que é busca híbrida?"
```

Se usar `-Config` no script, repita `--config config/lab.local.yaml` **antes** do
subcomando da CLI, ou defina `RAG_CONFIG` no `.env` para todos os comandos.

## Configuração dos modelos e módulos

[config/lab.yaml](config/lab.yaml) é a configuração compartilhada. Para ajustes pessoais:

```sh
cp config/lab.yaml config/lab.local.yaml
# No .env: RAG_CONFIG=config/lab.local.yaml
```

CLI e API consultam esse YAML em cada inicialização. No Compose, o arquivo selecionado
por `RAG_CONFIG` é montado em `/app/config/runtime.yaml`. Se editar os módulos, rode
`docker compose restart app`; se trocar o caminho em `RAG_CONFIG` ou variáveis do
`.env`, use `docker compose up -d --force-recreate app`. Não há recarga no meio de uma
consulta. CLI e PowerShell leem `.env` sem substituir variáveis já definidas no ambiente.

Exemplo de busca híbrida:

```yaml
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

Cada módulo desativado segue por **bypass** e aparece assim no `trace` da resposta.
Hybrid exige ao menos dois recuperadores; HyDE exige Embedding. Sem Hybrid, vários
recuperadores são intercalados por round-robin, com deduplicação e sem misturar suas
escalas de score. Todos os recuperadores podem ser desativados: nesse caso o sistema
informa que não há evidências e não gera uma resposta com o LLM.

Chat e embeddings são configurados **separadamente**. Para qualquer servidor que
exponha a API compatível de chat e/ou embeddings, altere a seção correspondente:

```yaml
llm:
  provider: openai
  base_url: http://localhost:1234/v1
  model: nome-do-modelo-servido
  api_key_env: LLM_API_KEY
  temperature: 0.0
  timeout: 120

embeddings:
  provider: openai
  base_url: http://localhost:1234/v1
  model: nome-do-modelo-de-embedding
  api_key_env: EMBEDDING_API_KEY
  dimensions: 768
  timeout: 120
```

Defina as chaves em `.env`. Se o servidor local não autenticar, configure uma string
convencional como `local` nas duas variáveis. Dentro do Docker, use um endereço
alcançável pelo container, como `http://host.docker.internal:1234/v1`.
O serviço `app` recebe `LLM_API_KEY` e `EMBEDDING_API_KEY`; se escolher outros nomes em
`api_key_env`, adicione-os ao `environment` do Compose. Não coloque segredos no YAML.

Ao trocar o modelo de embeddings, sua dimensão ou o endpoint, **reingira todos os
documentos**. O sistema rejeita consultas vetoriais a uma base com representações
ausentes ou incompatíveis. Ao mudar somente os módulos de consulta, não é necessário
reingerir, desde que os embeddings necessários já existam.

## API e inspeção do JSON

```sh
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"O que é RAG?","attributes":{"grupo":"G6","experimento":"baseline"}}'
```

`GET /health` verifica o banco; não testa o LLM. `GET /modules` informa os switches
carregados. `POST /query` devolve pergunta original, consultas derivadas, candidatos
por método, evidências, contexto efetivamente enviado, resposta e trace com tempos.
O campo `evidence` inclui fontes, posições, números de citação e scores por técnica.
O prompt solicita citações ao LLM, mas a correção dessas citações deve ser avaliada
pelos alunos: a geração não garante fidelidade factual.

## Organização e trabalho em grupos

- `src/rag_lab/config.py`: contrato e validação do YAML.
- `src/rag_lab/contracts.py`: contrato JSON com atributos extensíveis.
- `src/rag_lab/ingestion.py`: preparação dos documentos.
- `src/rag_lab/storage.py`: SQL e busca vetorial.
- `src/rag_lab/providers.py`: adaptadores LangChain para modelos.
- `src/rag_lab/modules/`: um arquivo por técnica/missão.
- `src/rag_lab/pipeline.py`: ordem, bypass, composição e geração.
- `scripts/start.ps1`: inicialização e preflight no Windows.
- `tests/`: contratos, composição, técnicas e integração com banco.

Consulte [arquitetura](docs/arquitetura.md) e [missões e protocolo experimental](docs/missoes.md).
Use branches por missão, combine alterações nos contratos antes de implementá-las e
entregue código, evidências da comparação e limitações da técnica. Preferências de
Codex, Claude, Cursor, Kimi e outras ferramentas, além do OpenSpec local, são ignoradas
pelo Git; instruções compartilhadas do laboratório ficam em `docs/`.

## Desenvolvimento e verificação

```sh
uv sync --frozen --extra dev --python 3.12
uv run rag-lab validate
uv run rag-lab doctor --offline
uv run ruff check .
uv run pytest
```

`doctor --offline` verifica Python e dependências; `doctor` também verifica os serviços
e modelos. O teste de integração é ignorado se `TEST_DATABASE_URL` não estiver definido.
Para executá-lo, use **um banco exclusivo de testes** com pgvector:

```sh
TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_test uv run pytest -m integration
```

Há uma workflow de CI com PostgreSQL/pgvector que executa também a integração. As
dependências Python são fixadas em `uv.lock`; ao alterá-las, atualize o lock com
`uv lock`. O PostgreSQL do Compose fica restrito a `localhost`, assim como a API.

## Limites desta primeira versão

O laboratório usa corpus pequeno: índices lexicais são reconstruídos em memória e a
busca vetorial é exata, sem índice ANN. Ingestão aceita apenas `.txt`/`.md` UTF-8;
PDF/OCR, autenticação, interface gráfica e operações distribuídas ficam para extensões.
A divisão em sentenças é uma heurística simples. Reranking usa sobreposição lexical e
compressão usa seleção extrativa; ambas são baselines para substituição pelos grupos.
A ingestão substitui atomicamente cada documento por caminho relativo à pasta de
entrada; use uma raiz estável e nomes distintos entre coleções. Arquivos removidos do
disco não são removidos automaticamente do banco. Não existe isolamento de coleções
nesta versão. Material recuperado é fornecido como dado ao LLM, mas o prompt sozinho
não elimina riscos de instruções maliciosas nos documentos.

Referências das integrações: [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama),
[ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai),
[OllamaEmbeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama) e
[pgvector](https://github.com/pgvector/pgvector).
