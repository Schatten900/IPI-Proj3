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

def aplicar_salt_pepper(img: np.ndarray, quantidade=0.008):

    resultado = img.copy()

    random = np.random.rand(*img.shape[:2])

    resultado[random < quantidade/2] = 0
    resultado[random > 1-quantidade/2] = 255

    return resultado

def aplicar_gaussian_noise(img: np.ndarray, mean = 0, sigma=25):

    # Cria o ruido gaussiano
    noise = np.random.normal(mean,sigma,img.shape).astype(np.float32)

    # Aplica o ruido a imagem original
    noisy_image = img.astype(np.float32) + noise

    # Fixa o intervalo entre 0 e 255
    noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)
    return noisy_image

def aplicar_ruido_imagem(img):
    gaussian_noisy = aplicar_gaussian_noise(img)
    return aplicar_salt_pepper(gaussian_noisy)
