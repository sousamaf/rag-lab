# Governança Documental, Segmentação e Pipeline de Ingestão

O pipeline de ingestão documental aurora-ingest padroniza a entrada de conhecimento na plataforma, aceitando exclusivamente arquivos de texto puro e Markdown codificados em UTF-8.
Formatos binários proprietários, planilhas ou documentos digitalizados que dependam de reconhecimento óptico de caracteres são bloqueados na triagem inicial e devem ser convertidos previamente.
A etapa de preparação segmenta o conteúdo em unidades correspondentes a sentenças individuais, aplicando uma divisão complementar sem sobreposição sempre que uma sentença ultrapassa o limite de 1000 caracteres.
O identificador único do documento é derivado do resumo SHA-256 do seu caminho relativo à pasta raiz de ingestão, enquanto o identificador de cada trecho combina o documento, a posição sequencial e o texto literal.
Em razão dessa identidade baseada no caminho relativo, a equipe de curadoria deve manter uma pasta raiz estável, pois renomear ou mover arquivos gera novos identificadores e duplica registros se o banco não for limpo.
A remoção física de um arquivo do diretório local não apaga automaticamente seus trechos já gravados no banco PostgreSQL, exigindo limpeza administrativa quando uma fonte é descontinuada.
Quando o recuperador de embeddings está habilitado no manifesto, o processo de ingestão envia os trechos ao modelo codificador em lotes máximos de 32 sentenças para controlar o uso de memória RAM.
Se o operador alterar o modelo de embeddings, o endereço do provedor ou a quantidade de dimensões no YAML, é obrigatório reingerir toda a coleção documental antes de executar novas buscas vetoriais.
Na etapa final de montagem do contexto enviado ao gerador, o sistema seleciona até a quantidade configurada de fontes principais e trunca o texto acumulado para respeitar o orçamento máximo de 12000 caracteres.
Apenas os trechos que efetivamente cabem no orçamento de caracteres recebem numeração sequencial de citação a partir de um e compõem a lista final de evidências do envelope de resposta.
