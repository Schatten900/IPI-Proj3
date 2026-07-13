from .utils import mostrar_imagem
import cv2
import numpy as np

#=====================================================
#   Modulo responsavel por aplicar o Canny classico
#=====================================================


#===================
#   Filtros
#===================

def aplicar_gaussiana(img : np.ndarray):
    # Funcao por reduzir o ruido aplicando filtro passa-baixas
    return cv2.GaussianBlur(img, (5,5), 0)


def aplicar_mediana(img : np.ndarray):
    # Funcao responsavel por aplicar a o filtro mediano para limpeza de paper-salt
    return cv2.medianBlur(img,5)


def aplicar_filtro_hibrido(img : np.ndarray):
    # Funcao por aplicar o os filtros adaptados mediana e gaussianos
    img_mediana = aplicar_mediana(img)
    img_filtrada = aplicar_gaussiana(img_mediana)

    return img_filtrada


#===================
#   Gradiente
#===================

def aplicar_gradiente_classico(img : np.ndarray):

    # Definindo os sobels classicos

    s1 = np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]
    ], dtype=np.float32)


    s2 = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], dtype=np.float32)


    # Derivada parcial de fs em y
    gx = cv2.filter2D(img, cv2.CV_32F, s2)

    gy = cv2.filter2D(img, cv2.CV_32F, s1)

    magnitude = np.sqrt(gx**2 + gy**2)

    direcao = np.arctan2(gy, gx)

    return magnitude,direcao

def aplicar_gradiente_adaptado(img : np.ndarray):
    # Funcao responsavel por achar o gradiente da imagem (Sobel 3x3) em 4 direcoes
    s1 = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ], dtype=np.float32)

    s2 = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], dtype=np.float32)

    s3 = np.array([
        [0,1,2],
        [-1,0,1],
        [-2,-1,0]
    ], dtype=np.float32)

    s4 = np.array([
        [-2,-1,0],
        [-1,0,1],
        [0,1,2]
    ], dtype=np.float32)

    g1 = cv2.filter2D(img, cv2.CV_32F, s1)
    g2 = cv2.filter2D(img, cv2.CV_32F, s2)
    g3 = cv2.filter2D(img, cv2.CV_32F, s3)
    g4 = cv2.filter2D(img, cv2.CV_32F, s4)

    magnitude = np.sqrt(g1**2 + g2**2 + g3**2 + g4**2)

    direcao = np.arctan2(g1, g2)

    return magnitude,direcao


#===================
#   Limiarização
#===================

def non_maximum_suppression(magnitude,direcao):
    # Funcao responsavel por transformar bordas grossas em finas

    altura, largura = magnitude.shape

    resultado = np.zeros((altura, largura), dtype=np.float32)

    # Radianos -> graus
    angulo = np.rad2deg(direcao)
    angulo[angulo < 0] += 180

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):

            q = 0
            r = 0

            # -----------------------------
            # 0°
            # -----------------------------
            if (0 <= angulo[i,j] < 22.5) or (157.5 <= angulo[i,j] <= 180):

                q = magnitude[i, j+1]
                r = magnitude[i, j-1]

            # -----------------------------
            # 45°
            # -----------------------------
            elif 22.5 <= angulo[i,j] < 67.5:

                q = magnitude[i-1, j+1]
                r = magnitude[i+1, j-1]

            # -----------------------------
            # 90°
            # -----------------------------
            elif 67.5 <= angulo[i,j] < 112.5:

                q = magnitude[i-1, j]
                r = magnitude[i+1, j]

            # -----------------------------
            # 135°
            # -----------------------------
            else:

                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            # Mantém apenas o máximo local
            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                resultado[i,j] = magnitude[i,j]
            else:
                resultado[i,j] = 0

    return resultado


def hysteresis(img : np.ndarray,forte=255,fraco=75):
    # Funcao responsavel por mantém bordas fracas conectadas às bordas fortes e remove bordas fracas isoladas.

    resultado = img.copy()

    # dimensoes da imagem
    altura, largura = img.shape

    for i in range(1,altura-1):
        for j in range(1, largura-1):
            
            # Percorre pelos pixels fracos 
            if resultado[i,j] == fraco:

                vizinhos = resultado[i-1:i+2, j-1:j+2]

                # Se os vizinhos sao fortes, o pixel fraco vira forte
                if np.any(vizinhos == forte):
                    resultado[i,j] = forte

                # Caso contrario é removido (fundo)
                else:
                    resultado[i,j] = 0  

    return resultado



def double_threshold(magnitude: np.ndarray,low=50, high=100):

    # Diferença pro adaptativo é que ele tenta criar esses valores
    forte = 255
    fraco = 75

    resultado = np.zeros_like(magnitude, dtype=np.uint8)

    # Bordas fortes
    resultado[magnitude >= high] = forte

    # Bordas fracas
    resultado[(magnitude >= low) & (magnitude < high)] = fraco

    return resultado


def calcular_threshold_global(img: np.ndarray, delta=1.0):

    # valores minimos e maximos de cinza da imagem
    hmin = np.min(img)
    hmax = np.max(img)

    # limiar inicial
    t0 = (hmin + hmax) / 2.0

    while True:

        # Conjunto dos valores acima e abaixo do limiar
        G1 = img[img > t0]
        G2 = img[img <= t0]

        # Evita divisão por zero
        if len(G1) == 0 or len(G2) == 0:
            break

        # valores medios dos conjuntos G1 e G2 
        m1 = np.mean(G1)
        m2 = np.mean(G2)

        t1 = (m1 + m2) / 2.0

        if abs(t1 - t0) < delta:
            break

        t0 = t1

    return t1

def double_threshold_adaptativo(magnitude: np.ndarray):

    high = calcular_threshold_global(magnitude)

    # Calcula novamente apenas no intervalo [0, high]
    low = calcular_threshold_global(magnitude[magnitude <= high])

    forte = 255
    fraco = 75

    resultado = np.zeros_like(magnitude, dtype=np.uint8)

    resultado[magnitude >= high] = forte

    resultado[(magnitude >= low) & (magnitude < high)] = fraco

    return resultado

#===================
#   Aplicacoes
#===================

def aplicar_canny_adaptado(img : np.ndarray):
    # Funcao responsavel por aplicar o filtro Canny do paper
    img_filtrada = aplicar_filtro_hibrido(img)

    mostrar_imagem(img_filtrada,"filtro hibrido")

    magnitude, direcao = aplicar_gradiente_adaptado(img_filtrada)

    magnitude = non_maximum_suppression(magnitude, direcao)

    img_limiarizada = double_threshold_adaptativo(magnitude)

    img_final = hysteresis(img_limiarizada)

    return img_final


def aplicar_canny_classico(img : np.ndarray):
    # Funcao por aplicar o filtro Canny tradicional a imagem
    img_gaussiana = aplicar_gaussiana(img)

    mostrar_imagem(img_gaussiana,"gaussiana_classico")

    # representa a força da borda 
    magnitude, direcao = aplicar_gradiente_classico(img_gaussiana)

    magnitude_nms = non_maximum_suppression(magnitude, direcao)

    img_limiarizada = double_threshold(magnitude_nms)

    img_final = hysteresis(img_limiarizada)

    return img_final