import zipfile
import os

# Nombre del archivo ZIP
zip_filename = "./DS-Lab4/lakes.zip"

# Carpeta de destino (puedes cambiarla)
output_folder = "./DS-Lab4/data"

# Crear la carpeta si no existe
os.makedirs(output_folder, exist_ok=True)

# Abrir y extraer el contenido
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

print(f"Archivo '{zip_filename}' descomprimido en la carpeta '{output_folder}'.")
