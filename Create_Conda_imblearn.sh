#!/bin/bash

set -e  # Detener script si hay error

ENV_NAME="imblearn"
ENV_FILE="environment.yml"

echo "========================================"
echo "Creando entorno conda: $ENV_NAME"
echo "========================================"

# 1. Verificar si environment.yml existe
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ ERROR: $ENV_FILE no encontrado"
    echo "   Genera primero: conda env export -n [tu_env] > $ENV_FILE"
    exit 1
fi

# 2. Crear entorno desde YAML
echo "📦 Creando entorno desde $ENV_FILE..."
conda env create -n "$ENV_NAME" -f "$ENV_FILE"

# 3. Activar entorno
echo "🚀 Activando entorno $ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 4. Registrar kernel para Jupyter (opcional)
read -p "¿Registrar kernel para Jupyter? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📓 Registrando kernel Jupyter..."
    python -m ipykernel install --user --name "$ENV_NAME" \
        --display-name "Python ($ENV_NAME)"
fi

# 5. Verificación
echo "✅ Entorno creado exitosamente"
echo "📊 Paquetes instalados:"
conda list | wc -l | xargs echo "   Total: "
conda list | head -10

echo ""
echo "========================================"
echo "PARA ACTIVAR MANUALMENTE:"
echo "   conda activate $ENV_NAME"
echo "========================================"