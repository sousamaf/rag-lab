# Implantação, Configuração Estrita e Diagnóstico Pré-Voo

O processo de implantação do Núcleo Aurora é regido pelo subsistema aurora-deploy e exige reprodutibilidade determinística das bibliotecas Python por meio de sincronização congelada a partir do arquivo de trava.
Na inicialização da interface de linha de comando ou do servidor web, o carregador lê o caminho definido na variável de ambiente AURORA_CONFIG_PATH ou recorre ao manifesto padrão do laboratório.
O validador de configuração aplica esquema estrito que proíbe campos desconhecidos, rejeitando imediatamente qualquer arquivo YAML que contenha erros de digitação ou opções não documentadas.
Valores booleanos de ativação de módulos devem ser declarados como tipos nativos verdadeiros ou falsos, sendo proibido o uso de textos entre aspas como sim, não, true ou false.
As regras de integridade cruzada do manifesto exigem que a ativação do módulo de fusão híbrida seja acompanhada por ao menos dois recuperadores habilitados simultaneamente.
Da mesma forma, o módulo de geração de documento hipotético só é aceito quando o recuperador vetorial por embeddings está ativo, e o número de candidatos por método deve ser maior ou igual ao limite final de fontes.
Antes de iniciar o servidor HTTP na porta padrão 8000, o script de partida executa o diagnóstico pré-voo completo para validar a versão do Python, as dependências, a porta de rede, o PostgreSQL 16 com extensão vetorial e a resposta dos modelos.
Não existe recarga dinâmica de configuração no meio do atendimento de requisições, portanto qualquer alteração nos switches do YAML exige o reinício explícito do processo da aplicação ou do contêiner.
A expansão de variáveis de ambiente dentro do YAML ocorre após a análise sintática estrutural, impedindo que caracteres especiais presentes em senhas de banco alterem a árvore do documento de configuração.
Em estações Windows nativas, o ambiente virtual deve utilizar Python na faixa da versão 3.11 até a 3.14, recomendando-se a versão 3.12 para paridade exata com a imagem oficial de contêiner.
