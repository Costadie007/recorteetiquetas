from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import io

from processamento import extrair_candidatos_etiqueta


app = FastAPI(
    title="API Recorte de Etiquetas",
    description="API para receber fotos e realizar o recorte automático das etiquetas.",
    version="1.0.0"
)


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
async def recortar_etiqueta(
    arquivo: UploadFile = File(...)
):
    """
    Recebe uma imagem, executa o algoritmo de recorte
    e devolve o melhor recorte encontrado.
    """

    # ---------------------------------------------------------
    # 1. VALIDAR TIPO DO ARQUIVO
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. LER A IMAGEM
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 3. EXECUTAR O ALGORITMO
    # ---------------------------------------------------------

    try:
        candidatos = extrair_candidatos_etiqueta(
            imagem_bytes
        )

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o processamento: {str(erro)}"
        )

    # ---------------------------------------------------------
    # 4. VERIFICAR SE ENCONTROU UMA ETIQUETA
    # ---------------------------------------------------------

    if not candidatos:
        raise HTTPException(
            status_code=404,
            detail="Nenhuma etiqueta foi encontrada na imagem."
        )

    # ---------------------------------------------------------
    # 5. PEGAR O MELHOR RESULTADO
    # ---------------------------------------------------------

    melhor_candidato = candidatos[0]

    imagem_recortada = melhor_candidato["imagem"]
    confianca = melhor_candidato["confianca"]

    # ---------------------------------------------------------
    # 6. CONVERTER RGB PARA PNG
    # ---------------------------------------------------------

    try:
        import cv2
        import numpy as np

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

    # ---------------------------------------------------------
    # 7. DEVOLVER A IMAGEM
    # ---------------------------------------------------------

    return Response(
        content=imagem_png,
        media_type="image/png",
        headers={
            "X-Confianca": str(confianca),
            "X-Quantidade-Candidatos": str(len(candidatos))
        }
    )
