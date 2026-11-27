# Filas de Mensageria e Processamento Assíncrono

O barramento assíncrono aurora-queue gerencia as solicitações de ingestão em lote e as notificações de auditoria por meio de fluxos persistentes baseados na tecnologia NATS JetStream.
Para impedir o reprocessamento acidental causado por instabilidades de rede, o corretor de mensagens aplica uma janela de desduplicação estrita de 120 segundos baseada no cabeçalho de identificação única do evento.
Cada envelope publicado na fila deve respeitar o limite físico máximo de 256 kilobytes, sendo que cargas maiores precisam ser particionadas pelo produtor antes da submissão.
Quando todos os trabalhadores de consumo estão ocupados e o buffer de espera atinge a capacidade máxima, o barramento rejeita novas publicações com o código de falha ERR-AUR-508.
Tarefas que falham durante a execução sofrem até 4 retentativas automáticas espaçadas por atraso exponencial iniciando em 5 segundos e dobrando a cada falha subsequente.
Após esgotar a quarta tentativa sem confirmação de sucesso, o corretor transfere a mensagem original e seu diagnóstico de erro para a fila de mensagens mortas denominada aurora.dlq.critico.
Os registros retidos na fila aurora.dlq.critico permanecem disponíveis para inspeção manual da equipe de engenharia durante 21 dias antes do expurgo definitivo do disco de mensageria.
Cada trabalhador de fila processa um único documento por vez e confirma o recebimento somente após o banco de dados confirmar a gravação durável em disco.
O monitoramento de atraso de consumo dispara um aviso preventivo sempre que a idade da mensagem mais antiga pendente na fila principal ultrapassa 180 segundos.
Filas dedicadas a telemetria de baixa prioridade utilizam política de descarte do item mais antigo quando o volume em disco atinge 80 por cento da partição alocada.
