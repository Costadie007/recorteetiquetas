import io
import os
from datetime import datetime

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from processamento import extrair_candidatos_etiqueta

app = FastAPI(
title="API Recorte de Etiquetas",
description="API para receber fotos, realizar o recorte automático das etiquetas e salvar no Google Drive.",
version="2.0.0"
)

# =========================================================

# CONFIGURAÇÕES DO GOOGLE DRIVE

# =========================================================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

def obter_servico_google_drive():
"""
Cria uma conexão autenticada com o Google Drive
utilizando o Refresh Token armazenado no Render.
"""

```
if not GOOGLE_CLIENT_ID:
    raise Exception("GOOGLE_CLIENT_ID não configurado.")

if not GOOGLE_CLIENT_SECRET:
    raise Exception("GOOGLE_CLIENT_SECRET não configurado.")

if not GOOGLE_REFRESH_TOKEN:
    raise Exception("GOOGLE_REFRESH_TOKEN não configurado.")

if not GOOGLE_DRIVE_FOLDER_ID:
    raise Exception("GOOGLE_DRIVE_FOLDER_ID não configurado.")

credenciais = Credentials(
    token=None,
    refresh_token=GOOGLE_REFRESH_TOKEN,
    token_uri="https://oauth2.googleapis.com/token",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    scopes=[
        "https://www.googleapis.com/auth/drive.file"
    ]
)

servico = build(
    "drive",
    "v3",
    credentials=credenciais
)

return servico
```

def salvar_no_google_drive(
imagem_png,
nome_arquivo
):
"""
Envia a imagem PNG para a pasta configurada
no Google Drive.
"""

```
servico = obter_servico_google_drive()

arquivo_metadata = {
    "name": nome_arquivo,
    "parents": [
        GOOGLE_DRIVE_FOLDER_ID
    ]
}

arquivo_em_memoria = io.BytesIO(imagem_png)

media = MediaIoBaseUpload(
    arquivo_em_memoria,
    mimetype="image/png",
    resumable=False
)

arquivo = (
    servico.files()
    .create(
        body=arquivo_metadata,
        media_body=media,
        fields="id,name,webViewLink"
    )
    .execute()
)

return arquivo
```

# =========================================================

# ROTAS BÁSICAS

# =========================================================

@app.get("/")
def inicio():
return {
"status": "online",
"mensagem": "API Recorte de Etiquetas funcionando"
}

@app.get("/api/health")
def health():
return {
"status": "ok"
}

# =========================================================

# RECORTE

# =========================================================

@app.post("/api/recortar")
async def recortar_etiqueta(
arquivo: UploadFile = File(...)
):
"""
Recebe uma imagem, executa o algoritmo de recorte,
salva o melhor recorte no Google Drive e retorna
informações sobre o arquivo salvo.
"""

```
tipos_permitidos = [
    "image/jpeg",
    "image/jpg",
    "image/png"
]

if arquivo.content_type not in tipos_permitidos:
    raise HTTPException(
        status_code=400,
        detail="Envie uma imagem JPG ou PNG."
    )

try:
    imagem_bytes = await arquivo.read()

except Exception:
    raise HTTPException(
        status_code=400,
        detail="Não foi possível ler a imagem enviada."
    )

if not imagem_bytes:
    raise HTTPException(
        status_code=400,
        detail="A imagem enviada está vazia."
    )

# -----------------------------------------------------
# PROCESSAMENTO DA IMAGEM
# -----------------------------------------------------

try:
    candidatos = extrair_candidatos_etiqueta(
        imagem_bytes
    )

except Exception as erro:
    raise HTTPException(
        status_code=500,
        detail=f"Erro durante o processamento: {str(erro)}"
    )

if not candidatos:
    raise HTTPException(
        status_code=404,
        detail="Nenhuma etiqueta foi encontrada na imagem."
    )

melhor_candidato = candidatos[0]

imagem_recortada = melhor_candidato["imagem"]
confianca = melhor_candidato["confianca"]

# -----------------------------------------------------
# CONVERTER O RECORTE PARA PNG
# -----------------------------------------------------

try:
    imagem_bgr = cv2.cvtColor(
        imagem_recortada,
        cv2.COLOR_RGB2BGR
    )

    sucesso, buffer = cv2.imencode(
        ".png",
        imagem_bgr
    )

    if not sucesso:
        raise Exception(
            "Não foi possível converter o recorte para PNG."
        )

    imagem_png = buffer.tobytes()

except Exception as erro:
    raise HTTPException(
        status_code=500,
        detail=f"Erro ao gerar imagem recortada: {str(erro)}"
    )

# -----------------------------------------------------
# NOME DO ARQUIVO
# -----------------------------------------------------

data_hora = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

nome_arquivo = (
    f"etiqueta_{data_hora}.png"
)

# -----------------------------------------------------
# GOOGLE DRIVE
# -----------------------------------------------------

try:
    arquivo_drive = salvar_no_google_drive(
        imagem_png,
        nome_arquivo
    )

except Exception as erro:
    raise HTTPException(
        status_code=500,
        detail=f"Erro ao salvar no Google Drive: {str(erro)}"
    )

# -----------------------------------------------------
# RESPOSTA
# -----------------------------------------------------

return {
    "status": "sucesso",
    "mensagem": "Etiqueta recortada e salva no Google Drive.",
    "nome_arquivo": arquivo_drive.get("name"),
    "arquivo_id": arquivo_drive.get("id"),
    "link": arquivo_drive.get("webViewLink"),
    "confianca": confianca,
    "quantidade_candidatos": len(candidatos)
}
```
