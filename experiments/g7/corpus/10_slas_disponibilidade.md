# Acordos de Nível de Serviço (SLA) e Disponibilidade

Os compromissos de qualidade operacional da Plataforma Núcleo Aurora estão formalizados na política POL-SLA-01, aplicável a todas as unidades acadêmicas subscritoras do plano Institucional.
A infraestrutura garante disponibilidade mensal mínima de 99,95 por cento, o que equivale a um orçamento máximo tolerado de 21 minutos de indisponibilidade não planejada por mês.
As janelas de manutenção preventiva programada ocorrem exclusivamente às terças-feiras, no horário entre 02h00 e 04h00 pelo fuso de Brasília, e não são descontadas do índice de disponibilidade mensal.
Para consultas padrão sem módulos de expansão prévia, o acordo estabelece que o percentil 95 do tempo total de atendimento não deve exceder 4500 milissegundos sob carga nominal.
Caso a disponibilidade mensal aferida fique abaixo da meta contratada ou o alerta de degradação de latência persista por mais de duas horas contínuas, a unidade recebe crédito compensatório de 12 por cento na fatura seguinte.
Incidentes classificados como severidade crítica S1 exigem tempo máximo de primeira resposta da equipe de plantão de 15 minutos e publicação de relatório pós-incidente em até 48 horas úteis.
Falhas originadas em provedores externos de modelos configurados pelo próprio usuário fora do cluster homologado são excluídas do cálculo de penalidades do acordo de nível de serviço.
Durante períodos de degradação parcial do motor de modelos de linguagem, o modo de contingência permite validar a configuração e inspecionar diagnósticos locais utilizando a flag de operação desconectada.
A aferição oficial de disponibilidade é realizada por sondas externas independentes que consultam o endpoint de saúde a cada 60 segundos a partir de três redes distintas.
Solicitações de abono ou crédito de nível de serviço devem ser protocoladas até o quinto dia útil do mês subsequente anexando os identificadores únicos das requisições afetadas.
