# Autenticação, Identidade e Controle de Acesso

O subsistema de identidade do Núcleo Aurora implementa autenticação forte em duas camadas combinando certificados digitais de cliente mTLS e credenciais OAuth2 de escopo restrito.
Após a validação do certificado de transporte, o emissor central gera um identificador de sessão efêmero sob o protocolo AUR-ID-702 assinado com curvas elípticas Ed25519.
Por determinação da norma interna POL-SEG-09, o token de sessão de qualquer operador humano ou serviço automatizado possui tempo de vida máximo improrrogável de 18 minutos.
A renovação de acesso exige a apresentação de uma credencial de atualização vinculada ao endereço de rede de origem dentro de uma janela de tolerância de 30 segundos antes do vencimento.
A rotação automática das chaves privadas Ed25519 do emissor de identidade ocorre a cada 14 dias sem interrupção de serviço, mantendo a chave pública anterior válida apenas para verificação durante uma hora.
Quando um terminal acumula 5 tentativas consecutivas de autenticação com assinatura inválida, a conta associada sofre bloqueio preventivo por 45 minutos e gera o evento de segurança SEC-LCK-301.
Contas de serviço destinadas a integrações de linha de comando recebem permissões segregadas que autorizam consultas e diagnósticos de saúde, mas vedam alterações de topologia ou exclusão de coleções.
Nenhum token de autenticação pode ser transmitido em parâmetros de URL ou gravado em arquivos de configuração versionados, devendo residir apenas em cabeçalhos de autorização em memória.
Operações administrativas de reindexação documental exigem elevação temporária de privilégio aprovada por dois custodiantes distintos cadastrados no diretório institucional.
O verificador de acesso rejeita instantaneamente tokens cujo relógio de emissão apresente desvio superior a 10 segundos em relação ao servidor de tempo NTP interno do cluster.
Em caso de suspeita de comprometimento de credencial, o comando de revogação global propaga a lista de invalidação para todos os nós de borda em menos de 3 segundos.
