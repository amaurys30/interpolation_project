✅ README.md — Proyecto: Software de Interpolación y Extrapolación 

📌 Software de Interpolación y Extrapolación 


Proyecto Final — Análisis de Técnicas Numéricas · 2025
Autores: Amaurys Castro – Daniel Jiménez
Institución: CECAR — Ingeniería

📘 Descripción general del proyecto

Este software implementa los métodos de interpolación y extrapolación vistos en clase.
Fue desarrollado como una aplicación interactiva usando Python + Dash, permitiendo:

Cargar datos desde un archivo CSV (columnas x, y).

Aplicar los métodos de interpolación vistos en clase:

Interpolación Lineal

Interpolación Cuadrática

Interpolación Cúbica

Lagrange grados 1, 2 y 3

Newton (diferencias divididas) grados 1, 2 y 3

Interpolación inversa (polinomio grado 3)

Generar gráficas individuales por método.

Generar una gráfica combinada (overlay) con todos los métodos.

Mostrar errores (RMSE, MAE, MaxErr, R²) para cada método.

Mostrar tablas con los valores estimados para todos los métodos.

Exportar los resultados a CSV.

El sistema permite analizar el comportamiento de los métodos, comparar resultados y visualizar el desempeño en tiempo real.

🛠 Requerimientos para ejecutar este proyecto

Para abrir este proyecto en cualquier computador después de descargarlo desde GitHub, necesitas instalar lo siguiente:

✔️ 1. Instalar Python 3.10 o superior

Descargar desde:
👉 https://www.python.org/downloads/

Durante la instalación marca la casilla:
✔️ "Add Python to PATH"

Recomendado: 
👉 Git

Descargar desde: https://git-scm.com/downloads

Este programa permite clonar el repositorio desde GitHub.

👉 Editor recomendado

Puedes usar cualquiera, pero recomendamos:

Visual Studio Code
https://code.visualstudio.com/

Extensiones recomendadas:

Python

Pylance

GitLens

✔️ 2. Instalar pip (si no viene instalado)

En consola (CMD o PowerShell):

python -m ensurepip --default-pip

✔️ 3. Crear un entorno virtual (recomendado)

En la carpeta del proyecto:

python -m venv venv


Activarlo:

En Windows:
venv\Scripts\activate

En Linux/Mac:
source venv/bin/activate

✔️ 4. Instalar dependencias

Ejecutar dentro del entorno virtual activado:

pip install -r requirements.txt


Si no tienes el archivo requirements.txt, puedes instalar todo con:

pip install dash plotly numpy pandas scipy

▶️ Cómo ejecutar el proyecto

Dentro del entorno virtual (activado), en la carpeta del proyecto:

python app.py


Luego abrir en el navegador:

http://127.0.0.1:8050/


La aplicación se abrirá automáticamente.

📂 Estructura del CSV requerido

Tu archivo CSV debe tener dos columnas en este orden:

x	y
0.0	1.5
1.2	2.8
2.1	3.4

...	...

IMPORTANTE:
✔️ La primera columna es x

✔️ La segunda columna es y

✔️ No importa si el CSV tiene encabezado o no (el software lo detecta)


Ejemplo de archivo válido:

x,y
0,1
1,2.5
2,3
3,5
4,7

📉 Funciones del software
🔹 1. Gráfica combinada (overlay)

Muestra todos los métodos seleccionados en una sola gráfica.

🔹 2. Gráficas individuales por método

Cada método genera:

✔️ Su gráfica
✔️ Una tabla con:

Valores evaluados (x)

Valores estimados (y_est)

🔹 3. Tabla de errores (panel derecho)

El sistema calcula automáticamente para cada método:

RMSE – Error cuadrático medio

MAE – Error absoluto medio

MaxErr – Error máximo

R² – Coeficiente de determinación

Nota adicional

🔹 4. Tabla combinada final

Muestra todos los valores estimados juntos:

Método	x_evaluado	y_estimado
🔹 5. Exportación a CSV

Exporta:

Tabla de errores

Tabla combinada de valores

En un solo archivo descargable.

🧮 Métodos incluidos
✔️ Interpolación

Lineal

Cuadrática

Cúbica

Lagrange (grado 1, 2 y 3)

Newton diferencias divididas (grado 1, 2 y 3)

✔️ Interpolación inversa

Polinomio grado 3

Devuelve los valores de x para un valor dado de y

✔️ Extrapolación

Todos los métodos permiten evaluar fuera del rango de los datos.

📦 Tecnologías utilizadas

Python 3

Dash

Plotly

NumPy

Pandas

SciPy

HTML/CSS (estilos integrados)

🚀 Cómo clonar y abrir el proyecto desde GitHub

Abrir una terminal

Clonar el repositorio:

git clone https://github.com/amaurys30/interpolation_project.git


Entrar al proyecto:

cd interpolation_project


Crear y activar entorno virtual:

python -m venv venv
venv\Scripts\activate   # Windows


Instalar dependencias:

pip install -r requirements.txt


🚀 Ejecutar:

En la terminal (con el entorno virtual activado):

python app.py


Luego abre en tu navegador:

http://127.0.0.1:8050

❌ Cómo salir del entorno virtual
Windows:
deactivate

Mac / Linux:
deactivate


✏️ Créditos

Proyecto desarrollado por:

Amaurys Castro

Daniel Jiménez
Corporación Universitaria del Caribe - CECAR — 2025

Docente: Carlos Cohen 
Asignatura: Análisis de Técnicas Numéricas

📄 Licencia

Este proyecto es de uso académico y educativo.
No se permite su uso comercial sin autorización de los autores.
