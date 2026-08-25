# 🌧️ Drought Forecast Binary Framework

Um framework para previsão de secas raras utilizando redes neurais espaço-temporais (ConvLSTM) com atenção temporal dual e transfer learning.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração](#configuração)
- [Como Usar](#como-usar)
- [Fluxo de Trabalho](#fluxo-de-trabalho)
- [Métricas](#métricas)
- [Resultados](#resultados)
- [Referências](#referências)

---

## 🎯 Visão Geral

Este framework foi desenvolvido para prever eventos de seca extrema (SPI ≤ -2.0) com 1 a 12 meses de antecedência, utilizando dados climáticos de reanálise para as cinco macro-regiões do Brasil.

### Características Principais

- **🧠 Arquitetura ConvLSTM com Atenção Temporal Dual**: Captura padrões espaço-temporais em dados climáticos
- **🔄 Transfer Learning**: Pré-treinamento com autoencoder para representações não supervisionadas
- **⚖️ Focal Loss com Balanceamento Dinâmico**: Lida com o desbalanceamento extremo das classes (prevalência ~2.9%)
- **📊 Calibração de Probabilidades**: Platt Scaling para probabilidades bem calibradas
- **🎯 Threshold Adaptativo**: Otimização automática baseada em MCC (Matthews Correlation Coefficient)
- **🌍 Processamento Espacial**: Pós-processamento morfológico para reduzir falsos positivos
- **🔁 Reprodutibilidade**: Sementes fixas e operações determinísticas

---

## 🏗️ Arquitetura

### Modelo Predictor

```
Input (B, T, C, H, W)
    ↓
ConvLSTM Encoder (3 camadas: 64 → 32 → 16)
    ↓
Dual Temporal Attention
    ├── Local Attention (curto prazo)
    ├── Global Attention (longo prazo)
    └── Adaptive Gate (combinação dinâmica)
    ↓
Prediction Decoder
    ↓
Logits (B, 1, H, W)
    ↓
Sigmoid → Probabilidades
```

### Autoencoder (Pré-treinamento)

```
Input Sequence (B, T, C, H, W)  # T = 12 meses
    ↓
ConvLSTM Encoder
    ↓
Dual Temporal Attention
    ↓
Temporal Decoder (reconstrução completa)
    ↓
Reconstructed Sequence (B, T, C_out, H, W)
```

---

## 📦 Instalação

### Requisitos

- Python 3.8+
- CUDA (opcional, para GPU)

### Passos

```bash
# Clone o repositório
git clone https://github.com/your-username/drought-forecast-binary.git
cd drought-forecast-binary

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Dependências

```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
rasterio>=1.3.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
openpyxl>=3.0.0
joblib>=1.1.0
```

---

## 📁 Estrutura do Projeto

```
drought_forecast_binary/
├── config/                              # Configuração centralizada
│   ├── __init__.py
│   ├── experiment_config.py             # Todas as configurações
│   └── paths.py                         # Gerenciamento de caminhos
│
├── data/                                # Manipulação de dados
│   ├── __init__.py
│   ├── loader.py                        # load_region_timeseries
│   ├── preprocessing.py                 # Downsample, máscara, normalização
│   ├── dataset.py                       # ClimateDataset
│   └── spi.py                           # Cálculo e cache do SPI
│
├── models/                              # Modelos neurais
│   ├── __init__.py
│   ├── convlstm_cell.py                 # Célula ConvLSTM
│   ├── encoder.py                       # ConvLSTMEncoder
│   ├── decoder.py                       # ReconstructionDecoder, PredictionDecoder
│   ├── attention.py                     # DualTemporalAttention
│   ├── autoencoder.py                   # ConvLSTMAutoencoder
│   └── predictor.py                     # ConvLSTMPredictor
│
├── training/                            # Lógica de treinamento
│   ├── __init__.py
│   ├── base.py                          # BaseTrainer
│   ├── autoencoder.py                   # AutoencoderTrainer
│   ├── predictor.py                     # PredictorTrainer
│   └── losses.py                        # FocalLoss, WeightedSmoothL1Loss
│
├── evaluation/                          # Ferramentas de avaliação
│   ├── __init__.py
│   ├── metrics.py                       # CSI, MCC, F1, etc.
│   ├── threshold.py                     # ThresholdOptimizer
│   └── calibration.py                   # PlattCalibrator, IsotonicCalibrator
│
├── utils/                               # Utilitários
│   ├── __init__.py
│   ├── logger.py                        # Logger colorido
│   ├── reproducibility.py               # Gerenciamento de sementes
│   ├── spatial.py                       # Operações morfológicas
│   └── geotiff.py                       # Exportação GeoTIFF
│
├── experiments/                         # Executores de experimentos
│   ├── __init__.py
│   ├── precompute_spi.py                # Pré-cálculo do SPI
│   ├── train_autoencoder.py             # Treinamento do autoencoder
│   ├── grid_search.py                   # Otimização de hiperparâmetros
│   └── analyze_variable_importance.py   # Análise de importância das variáveis
│
├── inference/                           # Módulo de inferência
│   ├── __init__.py
│   ├── predictor.py                     # InferencePredictor
│   ├── run.py                           # Executor de inferência
│   └── analyze.py                       # Análise de resultados
│
├── outputs/                             # Saídas (criado em tempo de execução)
│   └── {region}/                        # Por região (Sul, Norte, etc.)
│       ├── autoencoder/                 # Autoencoder treinado
│       ├── spi_cache/                   # Cache do SPI
│       ├── grid_search/                 # Resultados do grid search
│       ├── inferences/                  # Saídas de inferência
│       └── analysis/                    # Análises
│
├── requirements.txt                     # Dependências
├── main.py                              # Ponto de entrada principal
└── README.md                            # Documentação
```

---

## ⚙️ Configuração

Todas as configurações estão centralizadas em `config/experiment_config.py`:

### Principais Parâmetros

```python
@dataclass
class ExperimentConfig:
    # Dados
    data: DataConfig = field(default_factory=DataConfig)
    
    # SPI
    spi: SPIConfig = field(default_factory=SPIConfig)
    
    # Divisão temporal
    split: SplitConfig = field(default_factory=SplitConfig)
    
    # Arquitetura do modelo
    model: ModelArchConfig = field(default_factory=ModelArchConfig)
    
    # Treinamento
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Função de perda
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    
    # Grid search
    p_values: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6, 9, 12])
    
    # Reprodutibilidade
    random_seed: int = 42
```

### Variáveis Climáticas

O framework utiliza 7 variáveis do TerraClimate:

| Banda | Descrição | Unidade |
|-------|-----------|---------|
| `pr` | Precipitação | mm/mês |
| `pet` | Evapotranspiração Potencial | mm/mês |
| `soil` | Umidade do Solo | % |
| `srad` | Radiação Solar | W/m² |
| `vap` | Pressão de Vapor | kPa |
| `vs` | Velocidade do Vento | m/s |
| `tavg` | Temperatura Média | °C |

---

## 🚀 Como Usar

### Comando Principal

```bash
python3 main.py <comando> [opções]
```

### 1. Pré-calcular SPI

```bash
python3 main.py precompute-spi

# Para uma região específica
python3 main.py precompute-spi --region Sul
```

### 2. Treinar Autoencoder

```bash
# Treinamento padrão
python3 main.py train-ae

# Forçar retreinamento
python3 main.py train-ae --force
```

### 3. Grid Search

```bash
python3 main.py grid-search 

```

### 4. Inferência

```bash
# Inferência automática (usa melhor modelo do grid search)
python3 main.py inference

# Inferência com modelo específico
python3 main.py inference --region Sul --p 12 --q 1 --model-type pretrained

# Com threshold fixo
python3 main.py inference --region Sul --p 12 --q 1 --threshold 0.30
```

---

## 🔄 Fluxo de Trabalho

### 1. Pré-processamento
```
precompute_spi.py
├── Carrega rasters da região
├── Calcula SPI com rolling window (scale=3)
├── Analisa estatísticas
└── Salva em cache
```

### 2. Pré-treinamento (opcional)
```
train_autoencoder.py
├── Normaliza dados sazonalmente
├── Treina autoencoder com reconstrução de sequência
└── Salva modelo e normalizador
```

### 3. Grid Search
```
grid_search.py
├── Para cada (p, q) em p_values × q_values:
│   ├── Cria dataset com transfer learning
│   ├── Treina modelo com Focal Loss
│   ├── Otimiza threshold (MCC)
│   └── Salva checkpoint
└── Gera relatório com melhores configurações
```

### 4. Inferência
```
inference/run.py
├── Carrega melhor modelo
├── Carrega calibrador (se disponível)
├── Executa predições no período de teste
├── Otimiza threshold (fallback para validação)
└── Gera rasters GeoTIFF
```

### 5. Análise
```
inference/analyze.py
├── Calcula métricas por timestep
├── Gera gráficos de séries temporais
├── Mapas espaciais de frequência e viés
├── Curvas de calibração
└── Figuras de melhores/piores predições
```

---

## 📊 Métricas

### Principais Métricas

| Métrica | Descrição | Intervalo | Ideal |
|---------|-----------|-----------|-------|
| **MCC** | Matthews Correlation Coefficient | [-1, 1] | > 0.25 |
| **CSI** | Critical Success Index | [0, 1] | > 0.20 |
| **F1** | F1 Score | [0, 1] | > 0.35 |
| **Precision** | Proporção de acertos entre positivos | [0, 1] | > 0.30 |
| **Recall** | Proporção de eventos detectados | [0, 1] | > 0.60 |

### Métricas de Suporte

| Métrica | Descrição |
|---------|-----------|
| **FAR** | False Alarm Ratio (falsos alarmes) |
| **Bias** | Razão entre predições e observações |
| **ECE** | Expected Calibration Error |

---

## 📈 Resultados

### Melhor Configuração Encontrada (Sul, Seca Extrema)

| Parâmetro | Valor |
|-----------|-------|
| **p** | 18 |
| **q** | 1 |
| **MCC** | 0.4106 |
| **CSI** | 0.2908 |
| **Precision** | 40.0% |
| **Recall** | 64.7% |
| **F1** | 0.4946 |
| **Threshold** | 0.690 |

### Exemplo de Saída do Grid Search

```
======================================================================
                      🏆 BEST CONFIGURATION (MCC)
======================================================================
  Parameter                      Value
  ──────────────────────────────────────
  p                                 18
  q                                  1
  MCC                           0.4106
  CSI                           0.2908
  Threshold                      0.690
  Precision                     0.4002
  Recall                        0.6472
  F1                            0.4946
  Transfer Learning               True
  Augmentation                    True
──────────────────────────────────────
```

### Exemplo de Saída da Inferência

```
======================================================================
                   📊 RESUMO DA INFERÊNCIA
======================================================================
[INFO]   Região: Sul
[INFO]   SPI Threshold: -2.0
[INFO]   p: 18, q: 1
[INFO]   Modelo: pretrained
[INFO]   Período: 2022-01 a 2024-12
[INFO]   Amostras: 32
[INFO]   Threshold: 0.690
[INFO]   Calibração: ❌ Desativada
[INFO]   CSI: 0.2061
[INFO]   MCC: 0.1528
[INFO]   Precision: 0.2102
[INFO]   Recall: 0.9138
[INFO]   F1: 0.3418
```

### Resultados por Categoria

| Categoria | Threshold | Melhor (p,q) | MCC | CSI |
|-----------|-----------|--------------|-----|-----|
| **Moderada** | -1.0 | (3, 1) | 0.3869 | 0.4174 |
| **Severa** | -1.5 | (3, 1) | 0.3470 | 0.2961 |
| **Extrema** | -2.0 | (18, 1) | 0.4106 | 0.2908 |

---

