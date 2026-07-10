from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

class EduGPT:

    def __init__(self):
        self.name = "EduGPT"
        self.version = "0.0.2"

        # Carpeta donde estarán todos los libros
        self.books_path = BASE_DIR / "data" / "books"

        # Carpeta donde guardaremos el corpus unificado
        self.processed_path = BASE_DIR / "data" / "processed"

        # Archivo del corpus
        self.corpus_path = self.processed_path / "corpus.txt"

        self.vocabulary = set()
        self.total_words = 0

    def find_books(self):
        """
        Busca todos los archivos de texto
        dentro de la carpeta books.
        """

        books = list(self.books_path.glob("*.txt"))

        return books
    
    def show_books(self):
        books = self.find_books()
        print("\n========== LIBROS ENCONTRADOS ==========\n")
        if not books:
            print("No se encontró ningún libro.")
            return
        for index, book in enumerate(books, 1):
            print(f"{index}. {book.name}")
        print(f"\nTotal de libros: {len(books)}")

    def show_information(self):
        print("=" * 45)
        print(f"        {self.name} v0.0.1")
        print("=" * 45)
        print()

        print(f"Modelo: {self.name}")
        print("Estado: Activo")
        print(f"Corpus cargado: {self.total_words} palabras")
        print(f"Vocabulario: {len(self.vocabulary)} palabras")
        print()

        print("EduGPT:")
        print("Hola, todavía no sé nada.")
        print("Enséñame un libro.")