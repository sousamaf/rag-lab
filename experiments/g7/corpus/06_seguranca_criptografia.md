# Segurança da Informação, Rede e Criptografia

A arquitetura de defesa do Núcleo Aurora adota o princípio de privilégio mínimo e isolamento estrito de sub-redes entre os contêineres de aplicação e os motores de armazenamento.
Em conformidade com a diretriz normativa POL-SEG-14, o serviço de banco de dados PostgreSQL opera exclusivamente dentro da rede interna aurora_db_net e nunca publica a porta 5432 para interfaces externas.
Todos os volumes persistentes que abrigam fragmentos documentais e índices vetoriais são cifrados em repouso utilizando o algoritmo AES-256-GCM com chaves gerenciadas pelo cofre interno aurora-vault.
As chaves mestras de criptografia de disco armazenadas no aurora-vault passam por cerimônia de recifragem semestral conduzida pelo comitê de segurança da informação.
Para evitar vazamento de topologia e de segredos em ambientes de produção, qualquer exceção lançada por bibliotecas de clientes externos é interceptada na borda HTTP e convertida em erro genérico 503 sem exibir URLs ou chaves de API.
As credenciais de acesso a provedores externos de modelos de linguagem nunca podem constar no arquivo YAML principal, sendo referenciadas apenas pelo nome da variável de ambiente correspondente.
Como documentos ingeridos na base de conhecimento podem conter instruções maliciosas de injeção de comando, o prompt de sistema instrui explicitamente o modelo gerador a tratar os trechos recuperados como dados não confiáveis.
A interface HTTP pública do Núcleo Aurora disponibiliza apenas endpoints de leitura de saúde, listagem de módulos ativos e execução de consulta, vedando qualquer rota remota de ingestão por caminho de arquivo.
A ingestão de novos materiais ocorre exclusivamente por meio da interface de linha de comando executada localmente no servidor ou dentro do contêiner autorizado com montagem somente leitura da pasta de documentos.
Varreduras automatizadas de integridade verificam a cada hora se as permissões de sistema de arquivos dos certificados e variáveis de ambiente permanecem restritas ao usuário de serviço sem privilégio de superusuário.
