
class EduGPT:

    def __init__(self):
        self.name = "EduGPT"
        self.version = "0.0.2"

        self.vocabulary = set()
        self.total_words = 0

    def train(self):

        print()
        print("=" * 45)
        print("           ENTRENAMIENTO")
        print("=" * 45)
        print()

        print("Esta función estará disponible")
        print("en la siguiente misión.")
        print()
        print("Primero construiremos el")
        print("pipeline de entrenamiento.")

    def show_information(self):
        print()
        print("=" * 45)
        print(f"        {self.name} v{self.version}")
        print("=" * 45)

    def show_menu(self):

        self.show_information()

        print("1. Construir corpus")
        print("2. Entrenar modelo")
        print("3. Información del modelo")
        print("0. Salir")

        print()

        return input("Seleccione una opción: ")