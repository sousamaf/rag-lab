# Banco Vetorial e Indexação Semântica

A camada de persistência semântica do Núcleo Aurora opera sobre o motor aurora-vectordb, construído sobre PostgreSQL 16 com suporte nativo a operações algébricas em espaços de alta dimensão.
Cada unidade textual indexada armazena uma representação numérica densa de exatamente 768 coordenadas em ponto flutuante de precisão simples, acompanhada de uma chave criptográfica de configuração.
A assinatura de compatibilidade vetorial é calculada por um resumo SHA-256 que concatena o provedor, o endereço do serviço, o nome do modelo codificador e a dimensionalidade declarada.
Caso uma consulta encontre registros sem vetor ou com assinatura divergente da configuração ativa, o motor aborta a busca imediatamente com o código de erro ERR-AUR-417 e exige a reingestão integral da base.
Para coleções institucionais de até 50000 fragmentos textuais, o aurora-vectordb executa varredura exata por distância de cosseno sem índices aproximados ANN, assegurando revocação matemática perfeita na etapa vetorial.
O cálculo de relevância semântica converte a distância angular em pontuação de similaridade por meio da expressão um menos a distância de cosseno, ordenando os empatados pelo identificador hexadecimal do trecho.
Vetores compostos inteiramente por zeros ou que contenham valores não finitos como NaN ou infinito são rejeitados na camada de validação antes de qualquer comando de escrita no disco.
Durante a substituição de um documento existente, o banco adquire um bloqueio consultivo transacional derivado do identificador do documento para serializar escritas concorrentes e evitar estados parciais.
Se a geração dos vetores falhar no meio do processamento de um arquivo, a transação inteira sofre reversão automática e a versão anterior do documento permanece íntegra para os usuários.
As consultas de expansão de contexto por vizinhança utilizam um índice composto sobre o identificador do documento e a posição sequencial da sentença, recuperando trechos adjacentes em tempo submilissegundo.
