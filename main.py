
from src.utils import abrir_imagem, mostrar_imagem, aplicar_salt_pepper
from src.canny import aplicar_canny_adaptado, aplicar_canny_classico
#from src.morfologicas import aplicar_morfologicas

def aplicar_algoritmo():
    #img_cinza = abrir_imagem("img/man-in-cam.png",grayscale=True)
    #img_cinza = abrir_imagem("img/casa-segment.png",grayscale=True)
    img_cinza = abrir_imagem("img/flowers-seg.png",grayscale=True)

    imgRuidosa = aplicar_salt_pepper(img_cinza)
    mostrar_imagem(imgRuidosa,"imagem com ruido")

    # Comparacao com o classico
    imgCannyClassico = aplicar_canny_classico(imgRuidosa)
    mostrar_imagem(imgCannyClassico,"classico")

    # Comparacao com o adaptado
    imgCannyAdaptado = aplicar_canny_adaptado(imgRuidosa)
    mostrar_imagem(imgCannyAdaptado,"adaptado")

    # Aplica as morfologicas
    #imgMelhorada = aplicar_morfologicas(imgCanny)

    # Resultado esperado
    #mostrar_imagem(imgMelhorada)


if __name__ == "__main__":
    saiu = False
    print("Seja bem vindo ao trabalho final!")
    while (not saiu):
        print("1 - Rodar aplicacao")
        print("2 - Sair")
        escolha = input("Digite sua escolha: ")
        if escolha == '1':
            aplicar_algoritmo()
        if escolha == '2':
            print("Ate logo")
            saiu = True
        else:
            print("Escolha opcao valida")