# Processamento de Imagens com OpenCV e Python

Este repositório contém o trabalho final da matéria de Processamento de Imagens da UnB, desenvolvido em ambiente Linux. O projeto utiliza as bibliotecas **OpenCV** e **NumPy** para realizar filtragem morfológica e aplcação de filtros para segmentação.

---

## 📌 Descrição dos Trabalhos

### 🧠 Trabalho final: Aplicação da Segmentação Canny adaptado para imagens ruidosas
* **Objetivo:** Melhorar o algoritimo tradicional Canny por meio de aplicações morfologicas em imagens ruidosas.
* **Fluxo de Processamento:**
  1. Remoção de ruído utilizando filtros passa-baixas sequenciais (**Gaussiano** ).
  2. Aplicação do **Gradiente** a fim de detectar as bordas
  3. Transformações das bordas da imagem por meio do **Double Threshold**, **Hysteresis** e **Non Maximum Supression**
  4. Aplicação de operações morfológicas de **Abertura** e **Fechamento**.
  5. Análise da imagem obtida

---

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **OpenCV (`cv2`)**
* **NumPy**

---

## 🚀 Como Configurar e Executar (Linux)

O projeto possui uma interface de utilizador simples via terminal que permite alternar e testar ambos os trabalhos através de um menu interativo.

### 1. Clonar o Repositório
```bash
git clone https://github.com/Schatten900/IPI-Proj3
```
### 2. Configurar o ambiente (Venv)
Cria e ativa o ambiente virtual isolado para evitar conflitos de dependências:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências
Instalar os pacotes necessários por meio do requirements.txt
```bash
pip install -r requirements.txt
```

### 4. Executando a aplicação
```bash
python main.py
```