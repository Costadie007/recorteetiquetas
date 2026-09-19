import io
import os
import secrets
from datetime import datetime

import cv2
from fastapi import FastAPI, Request, HTTPException, Header, Depends
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

# Chave que o app precisa enviar no header X-API-Key
# para poder usar o /api/recortar.
API_KEY = os.getenv("API_KEY")

# Abaixo desse valor, o app deve avisar o usuário para
# conferir o recorte manualmente antes de confiar nele.
LIMIAR_CONFIANCA_BAIXA = 0.5


def validar_variaveis_obrigatorias():
    """Impede o serviço de subir se faltar alguma configuração essencial."""

    faltando = []

    if not GOOGLE_CLIENT_ID:
        faltando.append("GOOGLE_CLIENT_ID")

    if not GOOGLE_CLIENT_SECRET:
        faltando.append("GOOGLE_CLIENT_SECRET")

    if not GOOGLE_REFRESH_TOKEN:
        faltando.append("GOOGLE_REFRESH_TOKEN")

    if not GOOGLE_DRIVE_FOLDER_ID:
        faltando.append("GOOGLE_DRIVE_FOLDER_ID")

    if not API_KEY:
        faltando.append("API_KEY")

    if faltando:
        raise RuntimeError(
            "Variáveis de ambiente obrigatórias não configuradas: "
            + ", ".join(faltando)
        )


validar_variaveis_obrigatorias()


def verificar_api_key(
    x_api_key: str = Header(default=None)
):
    """Bloqueia o acesso se a chave enviada não bater com a configurada."""

    if not API_KEY:
        # Se a variável não foi configurada no Render, a rota
        # fica bloqueada por segurança, em vez de aberta por acidente.
        raise HTTPException(
            status_code=500,
            detail="API_KEY não configurada no servidor."
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Chave de API inválida ou ausente."
        )


# ============================================================
# PASTA TEMPORÁRIA PARA AS IMAGENS RECORTADAS
# ============================================================

PASTA_RECORTES = "/tmp/recortes"

os.makedirs(
    PASTA_RECORTES,
    exist_ok=True
)

# Quantidade máxima de recortes mantidos na pasta temporária.
# Os arquivos mais antigos são apagados automaticamente.
LIMITE_RECORTES = 20


def limpar_recortes_antigos():
    """Mantém apenas os recortes mais recentes na pasta temporária."""
    try:
        arquivos = [
            os.path.join(PASTA_RECORTES, nome)
            for nome in os.listdir(PASTA_RECORTES)
        ]
        arquivos = [c for c in arquivos if os.path.isfile(c)]
        arquivos.sort(key=os.path.getmtime, reverse=True)

        for caminho in arquivos[LIMITE_RECORTES:]:
            try:
                os.remove(caminho)
            except Exception:
                pass
    except Exception:
        pass


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
    imagem_arquivo,
    nome_arquivo,
    mimetype
):

    servico = obter_servico_google_drive()

    metadata = {
        "name": nome_arquivo,
        "parents": [
            GOOGLE_DRIVE_FOLDER_ID
        ]
    }

    media = MediaIoBaseUpload(
        io.BytesIO(imagem_arquivo),
        mimetype=mimetype,
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
        media_type="image/jpeg",
        filename=nome_arquivo
    )


# ============================================================
# PROCESSAMENTO DA ETIQUETA
# ============================================================

@app.post(
    "/api/recortar",
    dependencies=[Depends(verificar_api_key)]
)
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
    # CONVERTE PARA JPEG
    # --------------------------------------------------------

    sucesso, buffer = cv2.imencode(
        ".jpg",
        imagem_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )

    if not sucesso:

        raise HTTPException(
            status_code=500,
            detail="Não foi possível gerar o JPEG."
        )


    imagem_arquivo = buffer.tobytes()


    # --------------------------------------------------------
    # GERA NOME ÚNICO
    # --------------------------------------------------------

    nome_arquivo = (
        "etiqueta_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".jpg"
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
                imagem_arquivo
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

    # Remove recortes antigos para não acumular
    # arquivos indefinidamente na pasta temporária.
    limpar_recortes_antigos()


    # --------------------------------------------------------
    # SALVA A MESMA IMAGEM NO GOOGLE DRIVE
    # --------------------------------------------------------

    try:

        arquivo_drive = salvar_no_google_drive(
            imagem_arquivo,
            nome_arquivo,
            "image/jpeg"
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

        "metodo": candidatos[0].get(
            "metodo",
            "desconhecido"
        ),

        "confianca_baixa": (
            confianca < LIMIAR_CONFIANCA_BAIXA
            or candidatos[0].get("metodo") == "fallback"
        ),

        "quantidade_candidatos": len(
            candidatos
        )
    }
