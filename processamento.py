import gc
import os

import cv2
import numpy as np
import onnxruntime as ort

try:
    cv2.setNumThreads(1)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CAMINHO_MODELO = "best.onnx"
TAMANHO_ENTRADA = 416
CONF_MINIMA = 0.25
IOU_NMS = 0.50
MAX_DETECCOES = 3
MARGEM = 0.04

SESSAO = None
SESSAO_TENTADA = False


def carregar_sessao():
    """Carrega o modelo ONNX uma única vez e reaproveita."""
    global SESSAO, SESSAO_TENTADA

    if SESSAO_TENTADA:
        return SESSAO

    SESSAO_TENTADA = True

    if not os.path.exists(CAMINHO_MODELO):
        return None

    try:
        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = 1
        opcoes.inter_op_num_threads = 1
        opcoes.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        SESSAO = ort.InferenceSession(
            CAMINHO_MODELO,
            sess_options=opcoes,
            providers=["CPUExecutionProvider"],
        )
    except Exception:
        SESSAO = None

    return SESSAO


def letterbox(imagem_bgr, tamanho):
    """Redimensiona mantendo proporção e preenche o resto com cinza."""
    altura, largura = imagem_bgr.shape[:2]
    razao = min(tamanho / altura, tamanho / largura)

    nova_largura = max(1, int(round(largura * razao)))
    nova_altura = max(1, int(round(altura * razao)))

    redimensionada = cv2.resize(
        imagem_bgr,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_LINEAR,
    )

    tela = np.full((tamanho, tamanho, 3), 114, dtype=np.uint8)

    pad_x = (tamanho - nova_largura) // 2
    pad_y = (tamanho - nova_altura) // 2

    tela[
        pad_y:pad_y + nova_altura,
        pad_x:pad_x + nova_largura,
    ] = redimensionada

    del redimensionada
    return tela, razao, pad_x, pad_y


def detectar_com_onnx(img_bgr):
    """Retorna lista de (confianca, x1, y1, x2, y2) na escala original."""
    sessao = carregar_sessao()
    if sessao is None:
        return []

    altura_original, largura_original = img_bgr.shape[:2]

    entrada, razao, pad_x, pad_y = letterbox(img_bgr, TAMANHO_ENTRADA)

    tensor = cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB)
    tensor = tensor.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    tensor = np.ascontiguousarray(tensor)

    del entrada

    nome_entrada = sessao.get_inputs()[0].name
    saida = sessao.run(None, {nome_entrada: tensor})[0]

    del tensor

    # Saída do YOLO exportado: (1, 4 + num_classes, num_caixas)
    predicoes = np.squeeze(saida, axis=0).T
    del saida

    if predicoes.shape[1] < 5:
        return []

    scores = predicoes[:, 4:].max(axis=1)
    mascara = scores > CONF_MINIMA

    predicoes = predicoes[mascara]
    scores = scores[mascara]

    if predicoes.shape[0] == 0:
        return []

    cx = predicoes[:, 0]
    cy = predicoes[:, 1]
    larg = predicoes[:, 2]
    alt = predicoes[:, 3]

    x1 = cx - larg / 2
    y1 = cy - alt / 2

    caixas_nms = np.stack([x1, y1, larg, alt], axis=1)

    indices = cv2.dnn.NMSBoxes(
        caixas_nms.tolist(),
        scores.tolist(),
        CONF_MINIMA,
        IOU_NMS,
    )

    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()

    deteccoes = []

    for i in indices[:MAX_DETECCOES]:
        # Desfaz o letterbox e volta para a escala da foto original.
        cx1 = (x1[i] - pad_x) / razao
        cy1 = (y1[i] - pad_y) / razao
        cx2 = (x1[i] + larg[i] - pad_x) / razao
        cy2 = (y1[i] + alt[i] - pad_y) / razao

        cx1 = int(max(0, min(cx1, largura_original - 1)))
        cy1 = int(max(0, min(cy1, altura_original - 1)))
        cx2 = int(max(0, min(cx2, largura_original)))
        cy2 = int(max(0, min(cy2, altura_original)))

        deteccoes.append((float(scores[i]), cx1, cy1, cx2, cy2))

    return deteccoes


def recortar_com_margem(img_bgr, x1, y1, x2, y2, margem_relativa):
    altura_original, largura_original = img_bgr.shape[:2]

    largura_caixa = x2 - x1
    altura_caixa = y2 - y1

    if largura_caixa <= 0 or altura_caixa <= 0:
        return None

    margem_x = int(largura_caixa * margem_relativa)
    margem_y = int(altura_caixa * margem_relativa)

    xa = max(0, x1 - margem_x)
    ya = max(0, y1 - margem_y)
    xb = min(largura_original, x2 + margem_x)
    yb = min(altura_original, y2 + margem_y)

    recorte = img_bgr[ya:yb, xa:xb]

    if recorte.size == 0:
        return None

    return cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)


def fallback_opencv(img_bgr):
    """Usado só quando o modelo não encontra nada."""
    altura_original, largura_original = img_bgr.shape[:2]
    maior = max(altura_original, largura_original)

    escala = 1.0
    imagem = img_bgr

    if maior > 1600:
        escala = 1600 / maior
        imagem = cv2.resize(
            img_bgr,
            (
                max(1, int(largura_original * escala)),
                max(1, int(altura_original * escala)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    resultados = []

    try:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        cinza = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(cinza)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        gradiente = cv2.morphologyEx(cinza, cv2.MORPH_GRADIENT, kernel)

        _, binaria = cv2.threshold(
            gradiente, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        area_total = imagem.shape[0] * imagem.shape[1]

        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)

            if w <= 0 or h <= 0:
                continue

            proporcao = w / h
            area_relativa = (w * h) / area_total

            if not (1.5 <= proporcao <= 6.0):
                continue
            if not (0.008 <= area_relativa <= 0.70):
                continue

            if escala != 1.0:
                x = int(x / escala)
                y = int(y / escala)
                w = int(w / escala)
                h = int(h / escala)

            recorte = recortar_com_margem(
                img_bgr, x - 15, y - 15, x + w + 15, y + h + 15, 0.0
            )

            if recorte is not None:
                resultados.append(
                    {"imagem": recorte, "confianca": 0.5, "metodo": "fallback"}
                )

            if len(resultados) >= 3:
                break

    except Exception:
        pass

    finally:
        if imagem is not img_bgr:
            del imagem
        gc.collect()

    return resultados


def extrair_candidatos_etiqueta(imagem_bytes):
    array = np.frombuffer(imagem_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    del array

    if img_bgr is None:
        gc.collect()
        return []

    candidatos = []

    try:
        for conf, x1, y1, x2, y2 in detectar_com_onnx(img_bgr):
            recorte = recortar_com_margem(img_bgr, x1, y1, x2, y2, MARGEM)
            if recorte is not None:
                candidatos.append(
                    {"imagem": recorte, "confianca": conf, "metodo": "yolo"}
                )
    except Exception:
        pass

    if not candidatos:
        candidatos = fallback_opencv(img_bgr)

    del img_bgr
    gc.collect()

    candidatos.sort(key=lambda item: item["confianca"], reverse=True)
    return candidatos
