from .utils import abrir_imagem, mostrar_imagem
import cv2
import numpy as np

def aplicar_erosao(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    # Funcao responsavel por aplicar a erosao a imagem
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def aplicar_dilatacao(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    # Funcao responsavel por aplicar a dilatacao a imagem
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def aplicar_abertura(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    # Funcao responsavel por aplicar a abertura na imagem
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def aplicar_fechamento(img: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    # Funcao responsavel por aplicar o fechamento na imagem
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def criar_elementos_estruturantes():

    A1 = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ], dtype=np.uint8)


    A2 = np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0]
    ], dtype=np.uint8)


    A3 = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ], dtype=np.uint8)


    A4 = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=np.uint8)

    return [A1,A2,A3,A4]


def extrair_borda_morfologica(img, elemento):

    abertura = aplicar_abertura(img, elemento)

    fechamento = aplicar_fechamento(img, elemento)

    dil_abertura = aplicar_dilatacao(abertura, elemento)

    eros_fechamento = aplicar_erosao(fechamento, elemento)

    yd = cv2.subtract(dil_abertura, abertura)

    ye = cv2.subtract(fechamento, eros_fechamento)

    mapa = yd.astype(np.float32) + ye.astype(np.float32) + np.abs(yd.astype(np.float32)-ye.astype(np.float32))

    mapa = np.clip(mapa,0,255)

    return mapa.astype(np.uint8)

def extrair_quatro_mapas(img):

    elementos = criar_elementos_estruturantes()

    mapas = []

    for elemento in elementos:

        mapa = extrair_borda_morfologica(img, elemento)

        mapas.append(mapa)

    return mapas

def calcular_pesos_adaptativos(img):

    altura, largura = img.shape

    valores = np.zeros(4)

    for i in range(1,altura-1):

        for j in range(1,largura-1):

            janela = img[i-1:i+2,j-1:j+2]

            centro = janela[1,1]

            diferencas = np.abs(janela.astype(float)-float(centro))

            d1 = diferencas[0,0]
            d2 = diferencas[0,1]
            d3 = diferencas[0,2]

            d4 = diferencas[1,0]
            d6 = diferencas[1,2]

            d7 = diferencas[2,0]
            d8 = diferencas[2,1]
            d9 = diferencas[2,2]

            valores[0] += d1+d2+d3+d7+d8+d9

            valores[1] += d1+d2+d4+d6+d8+d9

            valores[2] += d1+d4+d7+d3+d6+d9

            valores[3] += d2+d3+d6+d4+d7+d8

    soma = np.sum(valores)

    if soma == 0:
        return np.ones(4)/4

    return valores/soma

def fundir_mapas(mapas,pesos):

    resultado = np.zeros_like(mapas[0],dtype=np.float32)


    for mapa,peso in zip(mapas,pesos):

        resultado += mapa.astype(np.float32)*peso


    resultado = np.clip(resultado,0,255)

    return resultado.astype(np.uint8)

def aplicar_morfologicas(img: np.ndarray) -> np.ndarray:

    mapas = extrair_quatro_mapas(img)

    pesos = calcular_pesos_adaptativos(img)

    resultado = fundir_mapas(mapas,pesos)

    return resultado