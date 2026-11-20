"""Protocolo primário, CONGELADO em 2026-11-18 (depois de ver os resultados; não é um pré-registro).

Uma só fonte para os parâmetros que definem os números principais da monografia. `tests/test_protocolo.py` confere os padrões das classes e
dos `argparse` do código contra estas constantes: se alguém mudar um padrão sem mudar o protocolo (ou o contrário), o teste falha. Mudar um
valor aqui invalida as tabelas de resultados e exige reexecutar os estudos afetados.
"""
from __future__ import annotations

# --- dados e divisão
SEEDS = (42, 7, 123)          # três divisões treino-teste por motor; as sementes NÃO são réplicas independentes (mesmos motores)
TEST_RATIO = 0.25             # fração de motores de teste em toda divisão por motor
RUL_CAP = 125                 # teto da RUL linear por partes
SEEDS_RUL = (42, 7, 123, 1, 2)  # benchmark de RUL: 5 sementes (as redes e o MiniRocket são determinísticos; o desvio entre sementes é variância de treino)

# --- janela saudável e deslocamento do motor
H_NATIVE = 50                 # motores nativos e camada de monitoramento
H_INJECTION = 25              # estudos de injeção (padrão de trabalho, não o ótimo: ver Seção 5.4)
W_SPLIT = 25                  # janela do deslocamento na análise de apoio "janela dividida"
ENGINE_OFFSET = True          # deslocamento do motor (por motor e regime, mín. 3 linhas) em TODOS os detectores
ENGINE_OFFSET_MEWMA_IN_RUL_INJECTION = False  # exceção deliberada: a injeção para a RUL (apêndice) usa a MEWMA sem o deslocamento

# --- detectores
EWMA_ALPHA = 0.10             # EWMA do |z| do T50 e dos sensores virtuais
THRESHOLD_SIGMA = 3.0         # portão por |z|
MEWMA_LAMBDA = 0.10
MEWMA_ALPHA = 0.0027          # nível do limite nominal do T²
PERSISTENCE_K = 2             # ciclos consecutivos acima do limite para o alarme (nos estudos; o padrão da classe VirtualSensor é 3 e só vale sem argumento)
MEWMA_CALIBRATION = "empirical"  # padrão da CLASSE e dos estudos de injeção (Seção 5.5 e a parte de injeção da Seção 5.4): limite do T² medido na janela
# EMENDA post hoc de 2026-11-18 (decisão do autor): nos motores nativos e na camada de monitoramento o limite é o SINTÉTICO, passado como argumento
# explícito em `monitoring/layer.py` (o padrão da classe não muda, porque mudaria as tabelas de injeção sem reexecutá-las). Validação só parcial
#; não é um resultado validado.
MEWMA_CALIBRATION_NATIVE = "synthetic"

# --- estudos de injeção
N_REPS_HARD = 2               # réplicas por motor nas falhas sorteadas (10 008 falhas)

# --- estatística
BOOTSTRAP_UNIT = "motor"      # reamostragem por motor receptor
HOLM_FAMILIES = {             # contrastes por família (Holm dentro de cada uma); fora delas, p sem correção, declarado
    "T50 sem portão - MEWMA com regime": 16,
    "regime na MEWMA (com - sem)": 12,
    "T50 sem portão - T50 bruto": 8,
}
