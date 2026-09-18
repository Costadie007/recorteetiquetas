
import io
import os
from datetime import datetime

import cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from processamento import extrair_candidatos_etiqueta


app = FastAPI(
    title="API Recorte de Etiquetas",
    version="3.0.0"
)


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")


# ============================================================
# PASTA TEMPORÁRIA PARA AS IMAGENS RECORTADAS
# ============================================================

PASTA_RECORTES = "/tmp/recortes"

os.makedirs(
    PASTA_RECORTES,
    exist_ok=True
)


# ============================================================
# GOOGLE DRIVE
# ============================================================

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


def salvar_no_google_drive(
    imagem_png,
    nome_arquivo
):

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


# ============================================================
# ROTAS BÁSICAS
# ============================================================

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

@app.get("/debug/memoria")
def debug_memoria():
    with open("/proc/self/status") as f:
        for linha in f:
            if linha.startswith("VmRSS:"):
                return {"rss_mb": round(int(linha.split()[1]) / 1024, 1)}
    return {"rss_mb": None}

# ============================================================
# ROTA PARA EXIBIR O RECORTE NO CELULAR
# ============================================================

@app.get("/api/recorte/{nome_arquivo}")
def obter_recorte(
    nome_arquivo: str
):

    caminho = os.path.join(
        PASTA_RECORTES,
        nome_arquivo
    )

    if not os.path.isfile(caminho):

        raise HTTPException(
            status_code=404,
            detail="Recorte não encontrado."
        )

    return FileResponse(
        caminho,
        media_type="image/png",
        filename=nome_arquivo
    )


# ============================================================
# PROCESSAMENTO DA ETIQUETA
# ============================================================

@app.post("/api/recortar")
async def recortar_etiqueta(
    request: Request
):

    # --------------------------------------------------------
    # RECEBE A FOTO ENVIADA PELO MIT APP INVENTOR
    # --------------------------------------------------------

    imagem_bytes = await request.body()

    if not imagem_bytes:

        raise HTTPException(
            status_code=400,
            detail="Nenhuma imagem foi recebida."
        )


    # --------------------------------------------------------
    # PROCESSA A IMAGEM COM O MESMO ALGORITMO EXISTENTE
    # --------------------------------------------------------

    candidatos = extrair_candidatos_etiqueta(
        imagem_bytes
    )

    if not candidatos:

        raise HTTPException(
            status_code=404,
            detail="Nenhuma etiqueta foi encontrada na imagem."
        )


    # --------------------------------------------------------
    # PEGA O MELHOR CANDIDATO
    # --------------------------------------------------------

    imagem_recortada = candidatos[0]["imagem"]

    confianca = candidatos[0]["confianca"]


    # --------------------------------------------------------
    # CONVERTE RGB PARA BGR
    # --------------------------------------------------------

    imagem_bgr = cv2.cvtColor(
        imagem_recortada,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # CONVERTE PARA PNG
    # --------------------------------------------------------

    sucesso, buffer = cv2.imencode(
        ".png",
        imagem_bgr
    )

    if not sucesso:

        raise HTTPException(
            status_code=500,
            detail="Não foi possível gerar o PNG."
        )


    imagem_png = buffer.tobytes()


    # --------------------------------------------------------
    # GERA NOME ÚNICO
    # --------------------------------------------------------

    nome_arquivo = (
        "etiqueta_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".png"
    )


    # --------------------------------------------------------
    # SALVA TEMPORARIAMENTE NO RENDER
    # PARA O CELULAR CONSEGUIR VISUALIZAR
    # --------------------------------------------------------

    caminho_recorte = os.path.join(
        PASTA_RECORTES,
        nome_arquivo
    )

    try:

        with open(
            caminho_recorte,
            "wb"
        ) as arquivo:

            arquivo.write(
                imagem_png
            )

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao salvar o recorte "
                "temporariamente: "
                + str(erro)
            )
        )


    # --------------------------------------------------------
    # SALVA A MESMA IMAGEM NO GOOGLE DRIVE
    # --------------------------------------------------------

    try:

        arquivo_drive = salvar_no_google_drive(
            imagem_png,
            nome_arquivo
        )

    except Exception as erro:

        # Se o Google Drive falhar, remove o
        # arquivo temporário para não deixar lixo.

        try:

            if os.path.exists(
                caminho_recorte
            ):
                os.remove(
                    caminho_recorte
                )

        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro ao salvar no Google Drive: "
                + str(erro)
            )
        )


    # --------------------------------------------------------
    # MONTA A URL QUE O CELULAR VAI USAR
    # --------------------------------------------------------

    url_recorte = str(
        request.base_url
    ).rstrip(
        "/"
    ) + "/api/recorte/" + nome_arquivo


    # --------------------------------------------------------
    # RETORNA RESULTADO PARA O MIT APP INVENTOR
    # --------------------------------------------------------

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

        "imagem_url": url_recorte,

        "confianca": confianca,

        "quantidade_candidatos": len(
            candidatos
        )
    }

