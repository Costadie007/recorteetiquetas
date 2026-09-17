import gc
import os

import cv2
import numpy as np


# ============================================================
# CONFIGURAÇÕES DE MEMÓRIA
# ============================================================

# Reduz a quantidade de threads usadas pelo OpenCV.
# Isso ajuda a evitar consumo desnecessário de RAM no Render.
try:
    cv2.setNumThreads(1)
except Exception:
    pass


# Limita também algumas bibliotecas matemáticas.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ============================================================
# IMPORTAÇÃO DO YOLO
# ============================================================

try:

    from ultralytics import YOLO

    YOLO_DISPONIVEL = True

except Exception:

    YOLO_DISPONIVEL = False


# ============================================================
# MODELO YOLO
# ============================================================

MODELO_YOLO = None
MODELO_CARREGADO = False


# ============================================================
# CONFIGURAÇÕES DO PROCESSAMENTO
# ============================================================

# Tamanho máximo da imagem usada para DETECÇÃO.
#
# A foto original pode ter milhares de pixels.
# Não precisamos mandar toda essa resolução para o YOLO.
#
# O recorte final continua sendo feito usando a imagem original.
MAX_DIMENSAO_DETECCAO = 1280


# Tamanho de entrada do YOLO.
#
# 416 reduz bastante o consumo de memória comparado
# com resoluções maiores.
TAMANHO_YOLO = 416


# Confiança mínima.
CONF_YOLO = 0.25


# IoU original do projeto.
IOU_YOLO = 0.50


# Quantidade máxima de detecções mantidas.
MAX_DETECCOES = 3


# ============================================================
# CARREGAR MODELO
# ============================================================

def carregar_modelo_yolo():

    global MODELO_YOLO
    global MODELO_CARREGADO

    # Se já carregamos o modelo, reutiliza.
    if MODELO_CARREGADO:

        return MODELO_YOLO

    MODELO_CARREGADO = True

    if not YOLO_DISPONIVEL:

        MODELO_YOLO = None

        return None

    try:

        caminho_modelo = "best.pt"

        if os.path.exists(caminho_modelo):

            MODELO_YOLO = YOLO(
                caminho_modelo
            )

        else:

            MODELO_YOLO = None

    except Exception:

        MODELO_YOLO = None

    return MODELO_YOLO


# ============================================================
# REDUZIR IMAGEM SOMENTE PARA DETECÇÃO
# ============================================================

def preparar_imagem_para_deteccao(
    imagem_bgr
):

    altura, largura = imagem_bgr.shape[:2]

    maior_dimensao = max(
        altura,
        largura
    )

    # Se já estiver pequena, não reduz.
    if maior_dimensao <= MAX_DIMENSAO_DETECCAO:

        return (
            imagem_bgr,
            1.0
        )

    escala = (
        MAX_DIMENSAO_DETECCAO
        / maior_dimensao
    )

    nova_largura = max(
        1,
        int(largura * escala)
    )

    nova_altura = max(
        1,
        int(altura * escala)
    )

    imagem_reduzida = cv2.resize(
        imagem_bgr,
        (
            nova_largura,
            nova_altura
        ),
        interpolation=cv2.INTER_AREA
    )

    return (
        imagem_reduzida,
        escala
    )


# ============================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================

def extrair_candidatos_etiqueta(
    imagem_bytes
):

    # --------------------------------------------------------
    # CONVERTE BYTES PARA ARRAY
    # --------------------------------------------------------

    array_imagem = np.frombuffer(
        imagem_bytes,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # DECODIFICA IMAGEM
    # --------------------------------------------------------

    img_bgr = cv2.imdecode(
        array_imagem,
        cv2.IMREAD_COLOR
    )

    # Libera imediatamente o array temporário.
    del array_imagem

    if img_bgr is None:

        gc.collect()

        return []


    # --------------------------------------------------------
    # DIMENSÕES DA IMAGEM ORIGINAL
    # --------------------------------------------------------

    altura_original, largura_original = (
        img_bgr.shape[:2]
    )


    # --------------------------------------------------------
    # LISTA DE RESULTADOS
    # --------------------------------------------------------

    candidatos = []


    # ========================================================
    # YOLO
    # ========================================================

    modelo = carregar_modelo_yolo()

    if modelo is not None:

        imagem_deteccao = None
        resultados = None

        try:

            # ------------------------------------------------
            # REDUZ SOMENTE PARA A DETECÇÃO
            # ------------------------------------------------

            (
                imagem_deteccao,
                escala
            ) = preparar_imagem_para_deteccao(
                img_bgr
            )


            # ------------------------------------------------
            # INFERÊNCIA YOLO
            # ------------------------------------------------

            resultados = modelo.predict(
                source=imagem_deteccao,

                conf=CONF_YOLO,

                iou=IOU_YOLO,

                imgsz=TAMANHO_YOLO,

                device="cpu",

                max_det=MAX_DETECCOES,

                verbose=False,

                stream=True
            )


            # ------------------------------------------------
            # PROCESSA RESULTADOS
            # ------------------------------------------------

            deteccoes = []

            for resultado in resultados:

                try:

                    if resultado.boxes is None:

                        continue

                    for box in resultado.boxes:

                        try:

                            conf = float(
                                box.conf[0]
                            )

                            coordenadas = (
                                box.xyxy[0]
                                .tolist()
                            )

                            x1, y1, x2, y2 = (
                                coordenadas
                            )

                            # --------------------------------
                            # VOLTA PARA COORDENADAS
                            # DA IMAGEM ORIGINAL
                            # --------------------------------

                            if escala != 1.0:

                                x1 = int(
                                    x1 / escala
                                )

                                y1 = int(
                                    y1 / escala
                                )

                                x2 = int(
                                    x2 / escala
                                )

                                y2 = int(
                                    y2 / escala
                                )

                            else:

                                x1 = int(x1)
                                y1 = int(y1)
                                x2 = int(x2)
                                y2 = int(y2)


                            # --------------------------------
                            # LIMITA A CAIXA À IMAGEM
                            # --------------------------------

                            x1 = max(
                                0,
                                min(
                                    x1,
                                    largura_original - 1
                                )
                            )

                            y1 = max(
                                0,
                                min(
                                    y1,
                                    altura_original - 1
                                )
                            )

                            x2 = max(
                                0,
                                min(
                                    x2,
                                    largura_original
                                )
                            )

                            y2 = max(
                                0,
                                min(
                                    y2,
                                    altura_original
                                )
                            )


                            deteccoes.append(
                                (
                                    conf,
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                )
                            )

                        except Exception:

                            continue

                except Exception:

                    continue


            # ------------------------------------------------
            # ORDENA PELA MAIOR CONFIANÇA
            # ------------------------------------------------

            deteccoes.sort(
                key=lambda item: item[0],
                reverse=True
            )


            # ------------------------------------------------
            # PEGA AS MELHORES DETECÇÕES
            # ------------------------------------------------

            for (
                conf,
                x1,
                y1,
                x2,
                y2
            ) in deteccoes[:MAX_DETECCOES]:

                largura_caixa = (
                    x2 - x1
                )

                altura_caixa = (
                    y2 - y1
                )

                if largura_caixa <= 0:

                    continue

                if altura_caixa <= 0:

                    continue


                # --------------------------------------------
                # MARGEM ORIGINAL DE 4%
                # --------------------------------------------

                margem_x = int(
                    largura_caixa * 0.04
                )

                margem_y = int(
                    altura_caixa * 0.04
                )


                x1_final = max(
                    0,
                    x1 - margem_x
                )

                y1_final = max(
                    0,
                    y1 - margem_y
                )

                x2_final = min(
                    largura_original,
                    x2 + margem_x
                )

                y2_final = min(
                    altura_original,
                    y2 + margem_y
                )


                # --------------------------------------------
                # RECORTE NA IMAGEM ORIGINAL
                # --------------------------------------------

                crop_bgr = img_bgr[
                    y1_final:y2_final,
                    x1_final:x2_final
                ]


                if crop_bgr.size == 0:

                    continue


                # --------------------------------------------
                # CONVERTE PARA RGB
                # --------------------------------------------

                crop_rgb = cv2.cvtColor(
                    crop_bgr,
                    cv2.COLOR_BGR2RGB
                )


                candidatos.append(
                    {
                        "imagem": crop_rgb,
                        "confianca": conf
                    }
                )


        except Exception:

            # Se o YOLO falhar, tentamos o fallback OpenCV.
            pass


        finally:

            # --------------------------------------------
            # LIBERA MEMÓRIA TEMPORÁRIA
            # --------------------------------------------

            try:

                del resultados

            except Exception:

                pass

            try:

                del imagem_deteccao

            except Exception:

                pass

            gc.collect()


    # ========================================================
    # FALLBACK OPENCV
    # ========================================================

    if not candidatos:

        gray = None
        clahe = None
        kernel = None
        gradient = None
        threshold = None
        kernel_close = None

        try:

            # ------------------------------------------------
            # ESCALA A IMAGEM PARA O FALLBACK
            # ------------------------------------------------

            imagem_fallback = img_bgr

            escala_fallback = 1.0

            maior_dimensao = max(
                altura_original,
                largura_original
            )

            if maior_dimensao > 1600:

                escala_fallback = (
                    1600
                    / maior_dimensao
                )

                nova_largura = max(
                    1,
                    int(
                        largura_original
                        * escala_fallback
                    )
                )

                nova_altura = max(
                    1,
                    int(
                        altura_original
                        * escala_fallback
                    )
                )

                imagem_fallback = cv2.resize(
                    img_bgr,
                    (
                        nova_largura,
                        nova_altura
                    ),
                    interpolation=cv2.INTER_AREA
                )


            # ------------------------------------------------
            # CINZA
            # ------------------------------------------------

            gray = cv2.cvtColor(
                imagem_fallback,
                cv2.COLOR_BGR2GRAY
            )


            # ------------------------------------------------
            # CLAHE
            # ------------------------------------------------

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            gray = clahe.apply(
                gray
            )


            # ------------------------------------------------
            # GRADIENTE MORFOLÓGICO
            # ------------------------------------------------

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (25, 7)
            )

            gradient = cv2.morphologyEx(
                gray,
                cv2.MORPH_GRADIENT,
                kernel
            )


            # ------------------------------------------------
            # OTSU
            # ------------------------------------------------

            _, threshold = cv2.threshold(
                gradient,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU
            )


            # ------------------------------------------------
            # FECHAMENTO
            # ------------------------------------------------

            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (25, 7)
            )

            threshold = cv2.morphologyEx(
                threshold,
                cv2.MORPH_CLOSE,
                kernel_close
            )


            # ------------------------------------------------
            # CONTORNOS
            # ------------------------------------------------

            contornos, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            area_imagem_fallback = (
                imagem_fallback.shape[1]
                * imagem_fallback.shape[0]
            )


            candidatos_fallback = []


            # ------------------------------------------------
            # ANALISA CONTORNOS
            # ------------------------------------------------

            for contorno in contornos:

                x, y, w, h = (
                    cv2.boundingRect(
                        contorno
                    )
                )

                if h <= 0:

                    continue

                if w <= 0:

                    continue


                proporcao = (
                    w / h
                )

                area = w * h

                proporcao_area = (
                    area
                    / area_imagem_fallback
                )


                # Mesmos filtros do algoritmo original.

                if proporcao < 1.5:

                    continue

                if proporcao > 6.0:

                    continue

                if proporcao_area < 0.008:

                    continue

                if proporcao_area > 0.70:

                    continue


                # ------------------------------------------------
                # CONVERTE COORDENADAS PARA ORIGINAL
                # ------------------------------------------------

                if escala_fallback != 1.0:

                    x = int(
                        x / escala_fallback
                    )

                    y = int(
                        y / escala_fallback
                    )

                    w = int(
                        w / escala_fallback
                    )

                    h = int(
                        h / escala_fallback
                    )


                # ------------------------------------------------
                # MARGEM ORIGINAL DE 15 PX
                # ------------------------------------------------

                x1 = max(
                    0,
                    x - 15
                )

                y1 = max(
                    0,
                    y - 15
                )

                x2 = min(
                    largura_original,
                    x + w + 15
                )

                y2 = min(
                    altura_original,
                    y + h + 15
                )


                crop_bgr = img_bgr[
                    y1:y2,
                    x1:x2
                ]


                if crop_bgr.size == 0:

                    continue


                crop_rgb = cv2.cvtColor(
                    crop_bgr,
                    cv2.COLOR_BGR2RGB
                )


                candidatos_fallback.append(
                    {
                        "imagem": crop_rgb,
                        "confianca": 0.5
                    }
                )


                # Não precisamos guardar dezenas
                # de candidatos no fallback.
                if len(
                    candidatos_fallback
                ) >= 3:

                    break


            candidatos.extend(
                candidatos_fallback
            )


        except Exception:

            pass


        finally:

            # ------------------------------------------------
            # LIBERA MEMÓRIA DO FALLBACK
            # ------------------------------------------------

            try:
                del gray
            except Exception:
                pass

            try:
                del clahe
            except Exception:
                pass

            try:
                del kernel
            except Exception:
                pass

            try:
                del gradient
            except Exception:
                pass

            try:
                del threshold
            except Exception:
                pass

            try:
                del kernel_close
            except Exception:
                pass

            try:
                del imagem_fallback
            except Exception:
                pass

            gc.collect()


    # ========================================================
    # ORDENA RESULTADOS
    # ========================================================

    candidatos.sort(
        key=lambda item: item["confianca"],
        reverse=True
    )


    # ========================================================
    # LIBERA MEMÓRIA
    # ========================================================

    gc.collect()


    return candidatos