
from src.utils import abrir_imagem, mostrar_imagem, aplicar_ruido_imagem
from src.canny import aplicar_canny_adaptado, aplicar_canny_classico
from src.morfologicas import aplicar_morfologicas
import numpy as np


def fundir_canny_morfologia(canny, morfologia, alpha=0.5):

    if canny.shape != morfologia.shape:
        raise ValueError("As imagens precisam ter o mesmo tamanho")

    F1 = canny.astype(np.float32)
    F2 = morfologia.astype(np.float32)

    F3 = np.maximum(F1, F2)

    F4 = alpha * F1 + (1-alpha) * F2

    F5 = F3 + F4

    F5 = np.clip(F5,0,255)

    return F5.astype(np.uint8)

def aplicar_algoritmo():
    #img_cinza = abrir_imagem("img/man-in-cam.png",grayscale=True)
    img_cinza = abrir_imagem("img/casa-segment.png",grayscale=True)
    #img_cinza = abrir_imagem("img/flowers-seg.png",grayscale=True)

    imgRuidosa = aplicar_ruido_imagem(img_cinza)
    mostrar_imagem(imgRuidosa,"imagem com ruido")

    # Comparacao com o classico
    imgCannyClassico = aplicar_canny_classico(imgRuidosa)
    mostrar_imagem(imgCannyClassico,"classico")

    # Comparacao com o adaptado
    imgCannyAdaptado = aplicar_canny_adaptado(imgRuidosa)
    mostrar_imagem(imgCannyAdaptado,"adaptado")

    imgMorfologica = aplicar_morfologicas(imgRuidosa)

    mostrar_imagem(imgMorfologica,"morfologia")


    imgFinal = fundir_canny_morfologia(imgCannyAdaptado, imgMorfologica)

    mostrar_imagem(imgFinal,"resultado final")

if __name__ == "__main__":
    saiu = False
    print("Seja bem vindo ao trabalho final!")
    while (not saiu):
        print("1 - Rodar aplicacao")
        print("2 - Sair")
        escolha = input("Digite sua escolha: ")
        if escolha == '1':
            aplicar_algoritmo()
        elif escolha == '2':
            print("Ate logo")
            saiu = True
        else:
            print("Escolha opcao valida")