# Faturamento, Cotas de Consumo e Bilhetagem

O módulo de controle econômico aurora-billing contabiliza o uso computacional de cada departamento acadêmico a partir da quantidade de tokens processados nos modelos de geração e de codificação vetorial.
Cada unidade departamental recebe uma franquia mensal padrão de 4500000 tokens de inferência, renovada automaticamente no primeiro minuto do dia primeiro de cada mês civil.
Para proteger o cluster contra rajadas abusivas de um único laboratório, o limitador de taxa aplica uma janela móvel de tarifação de 60 segundos com teto de 30000 tokens por minuto.
Quando uma unidade esgota sua franquia mensal ou ultrapassa o teto da janela móvel, as novas requisições de geração são recusadas com o código de bloqueio orçamentário ERR-AUR-429.
O sistema emite um alarme preventivo para o gestor financeiro sempre que o consumo acumulado do mês atinge 85 por cento da cota departamental alocada.
Consultas que resultam em ausência de evidências na base documental e não acionam o modelo gerador final são isentas de tarifação de síntese, debitando apenas o custo de codificação da busca vetorial.
Departamentos com autorização de uso excedente pagam uma tarifa complementar calculada por lote indivisível de mil tokens adicionais processados acima da franquia básica.
Módulos opcionais de expansão de consulta que invocam o modelo de linguagem antes da etapa de recuperação têm seu consumo somado integralmente ao bilhete da requisição principal.
O relatório consolidado de bilhetagem é fechado diariamente às 23h50 e exportado em formato tabular assinado para conciliação no sistema contábil da instituição.
Testes automatizados de diagnóstico executados pelo utilitário de verificação pré-voo consomem tokens reais do endpoint configurado e são contabilizados na cota de manutenção da infraestrutura.
