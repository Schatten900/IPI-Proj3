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


def aplicar_morfologicas(img: np.ndarray) -> np.ndarray:
    # Funcao responsavel por aplicar a deteccao de borda morfologica multidirecional.
    pass