# Backup, Salvaguarda e Recuperação de Desastres

O plano de continuidade operacional do Núcleo Aurora é executado pelo serviço aurora-backup, que coordena cópias de segurança consistentes da base relacional e dos índices vetoriais.
Capturas incrementais das transações do PostgreSQL são realizadas automaticamente a cada 6 horas, enquanto uma imagem completa consolidada é gerada todos os domingos às 01h00.
Os artefatos de salvaguarda são transferidos para armazenamento objeto externo configurado em modo imutável WORM, onde permanecem sob retenção obrigatória por 90 dias.
A arquitetura de replicação assíncrona garante um objetivo de ponto de recuperação RPO máximo de 15 minutos, limitando a perda potencial de ingestões recentes em caso de colapso do datacenter primário.
Em cenários de falha catastrófica regional, o procedimento automatizado de promoção da instância secundária assegura um objetivo de tempo de recuperação RTO de até 40 minutos.
Para garantir a integridade dos dados durante reingestões cotidianas antes mesmo do backup externo, cada documento é atualizado em uma transação atômica única no banco de dados.
Dessa forma, se ocorrer interrupção elétrica ou falha no provedor de embeddings durante a indexação, nenhuma metade de arquivo fica gravada e a versão anterior permanece disponível sem corrupção.
Trimestralmente, a equipe de confiabilidade executa um exercício obrigatório de restauração limpa em ambiente isolado sob o protocolo DR-SIM-04 para aferir a validade criptográfica dos arquivos de backup.
Os volumes de dados locais do Docker preservam a base entre reinicializações normais dos contêineres, desde que o operador não invoque o comando de remoção explícita de volumes.
Cópias dos manifestos YAML utilizados nos experimentos científicos devem ser arquivadas junto aos resultados brutos para permitir a reprodução exata do comportamento do pipeline no futuro.
