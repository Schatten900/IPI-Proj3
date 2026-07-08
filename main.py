
from src.utils import abrir_imagem, mostrar_imagem
from src.canny import aplicar_canny
from src.morfologicas import aplicar_morfologicas
import cv2

def aplicar_algoritmo():
    img = abrir_imagem("img/",grayscale=True)
    imgCanny = aplicar_canny(img)

    # Aplica as morfologicas
    imgMelhorada = aplicar_morfologicas(imgCanny)

    # Resultado esperado
    mostrar_imagem(imgMelhorada)


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
            escolha = True
        else:
            print("Escolha opcao valida")