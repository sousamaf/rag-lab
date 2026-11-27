# Arquitetura Geral e Topologia do Núcleo Aurora

A Plataforma Núcleo Aurora é um ecossistema corporativo de recuperação e síntese documental projetado para operar em ambientes acadêmicos e governamentais de alta criticidade.
Toda a comunicação externa ingressa exclusivamente por um proxy reverso Envoy operando na porta 9443 com terminação criptográfica mútua e inspeção de cabeçalhos.
O serviço coordenador central denomina-se aurora-core e executa o roteamento determinístico das requisições entre os subsistemas de preparação, busca e síntese.
Cada nó de borda do cluster suporta o limite estrito de 380 conexões simultâneas ativas antes de acionar o mecanismo de contrapressão HTTP 429.
Internamente, os microsserviços trocam mensagens estruturadas em JSON sobre canais internos de baixa latência, garantindo que nenhum objeto proprietário de biblioteca externa atravesse as fronteiras de processo.
A cada etapa ativa do fluxo de processamento, o coordenador aurora-core serializa e revalida o contrato de dados completo para impedir mutações estruturais acidentais ou vazamento de descritores de rede.
Os módulos desativados no manifesto operacional não são instanciados em memória e recebem apenas um registro de passagem direta na trilha de execução da consulta.
A topologia de implantação separa fisicamente a sub-rede de aplicação da sub-rede de persistência para que somente o coordenador autorizado alcance os motores de armazenamento.
Em cenários de pico de tráfego, o balanceador interno prioriza consultas de leitura curta sobre tarefas administrativas de indexação em lote por meio de filas ponderadas.
O estado operacional de uma requisição é inteiramente contido no envelope de trânsito, pois os trabalhadores do aurora-core são estritamente apátridas e não mantêm sessões locais em memória RAM.
Quando todos os motores de recuperação de uma rota retornam conjuntos vazios, o coordenador aborta a chamada ao gerador textual e devolve imediatamente uma mensagem padronizada de ausência de lastro documental.
A descoberta de serviços dentro do cluster utiliza registros DNS estáticos resolvidos pelo daemon interno na porta 5353, eliminando dependências de registradores externos durante falhas de conectividade WAN.
