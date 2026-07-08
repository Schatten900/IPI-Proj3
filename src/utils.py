import cv2
import numpy as np
import os

def abrir_imagem(path : str, grayscale : bool = False) -> np.ndarray:
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path,cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Erro ao tentar ler  imagem: {path}")
    
    return img

def salvar_imagem(path : str,img : np.ndarray) -> None:
    os.makedirs(os.path.dirname(path),exist_ok=True)
    cv2.imwrite(path,img)


def mostrar_imagem(img : np.ndarray, title : str) -> None:
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def separar_canais(img : np.ndarray) -> list:
    return cv2.split(img)

def juntar_canais(channels : list) -> np.ndarray:
    return cv2.merge(channels)