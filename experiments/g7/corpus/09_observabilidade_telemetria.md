# Observabilidade, Telemetria e Monitoramento de Latência

O subsistema aurora-telemetry instrumenta todas as etapas do pipeline de recuperação e geração com cronômetros de alta resolução baseados em contadores monotônicos de desempenho.
As métricas coletadas e os rastros distribuídos são exportados continuamente via protocolo OpenTelemetry gRPC na porta 4317 para o coletor central da infraestrutura.
Nos servidores locais, os arquivos de registro operacional sofrem rotação automática sempre que atingem o tamanho de 500 megabytes, mantendo até dez arquivos históricos compactados.
O painel de monitoramento em tempo real calcula as estatísticas de percentil p95 e p99 de tempo de resposta utilizando uma janela deslizante de agregação de 30 segundos.
Quando a duração isolada da etapa de geração textual no modelo de linguagem ultrapassa 4500 milissegundos em três amostras seguidas, o sistema dispara o alerta operacional ALR-LAT-901.
Cada entrada do rastro de execução no envelope JSON informa o nome do módulo, o status executado ou ignorado, a duração em milissegundos com três casas decimais, a contagem de evidências e o total de candidatos.
Para diagnóstico fino da fusão híbrida, a telemetria avançada registra em atributo dedicado a estratégia ativa, os pesos aplicados, a sobreposição entre recuperadores e as contribuições numéricas de cada ranking.
O indicador booleano de empate de fronteira sinaliza quando o último item incluído no corte de contexto empatou exatamente em pontuação com o primeiro item excluído do orçamento.
Enquanto consultas normais passam por amostragem estatística de 10 por cento no coletor de rastros, requisições que resultam em erro HTTP 503 ou violação de tempo limite são gravadas de forma integral.
A verificação rápida de integridade na rota HTTP de saúde executa apenas uma consulta leve de conectividade ao banco PostgreSQL, evitando acionar o modelo de linguagem para não gerar custo nem latência artificial.
