import io
import os
from datetime import datetime

import cv2
from fastapi import FastAPI, Request, HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from processamento import extrair_candidatos_etiqueta


app = FastAPI(
    title="API Recorte de Etiquetas",
    version="2.1.0"
)


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")


def obter_servico_google_drive():

    if not all([
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        GOOGLE_REFRESH_TOKEN,
        GOOGLE_DRIVE_FOLDER_ID
    ]):
        raise Exception(
            "As variáveis do Google Drive não estão configuradas no Render."
        )

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


def salvar_no_google_drive(imagem_png, nome_arquivo):

    servico = obter_servico_google_drive()

    metadata = {
        "name": nome_arquivo,
        "parents": [
            GOOGLE_DRIVE_FOLDER_ID
        ]
    }

    media = MediaIoBaseUpload(
        io.BytesIO(imagem_png),
        mimetype="image/png",
        resumable=False
    )

    arquivo = servico.files().create(
        body=metadata,
        media_body=media,
        fields="id,name,webViewLink"
    ).execute()

    return arquivo


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


@app.post("/api/recortar")
async def recortar_etiqueta(request: Request):

    imagem_bytes = await request.body()

    if not imagem_bytes:

        raise HTTPException(
            status_code=400,
            detail="Nenhuma imagem foi recebida."
        )

    candidatos = extrair_candidatos_etiqueta(
        imagem_bytes
    )

    if not candidatos:

        raise HTTPException(
            status_code=404,
            detail="Nenhuma etiqueta foi encontrada na imagem."
        )

    imagem_recortada = candidatos[0]["imagem"]

    confianca = candidatos[0]["confianca"]

    imagem_bgr = cv2.cvtColor(
        imagem_recortada,
        cv2.COLOR_RGB2BGR
    )

    sucesso, buffer = cv2.imencode(
        ".png",
        imagem_bgr
    )

    if not sucesso:

        raise HTTPException(
            status_code=500,
            detail="Não foi possível gerar o PNG."
        )

    nome_arquivo = (
        "etiqueta_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".png"
    )

    try:

        arquivo_drive = salvar_no_google_drive(
            buffer.tobytes(),
            nome_arquivo
        )

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao salvar no Google Drive: "
                + str(erro)
            )
        )

    return {
        "status": "sucesso",
        "mensagem": (
            "Etiqueta recortada e salva "
            "no Google Drive."
        ),
        "nome_arquivo": arquivo_drive.get(
            "name"
        ),
        "arquivo_id": arquivo_drive.get(
            "id"
        ),
        "link": arquivo_drive.get(
            "webViewLink"
        ),
        "confianca": confianca,
        "quantidade_candidatos": len(
            candidatos
        )
    }