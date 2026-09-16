import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_DISPONIVEL = True
except Exception:
    YOLO_DISPONIVEL = False


def carregar_modelo_yolo():
    """
    Carrega o modelo YOLO utilizado pelo projeto.
    """
    if not YOLO_DISPONIVEL:
        return None

    try:
        import os

        if os.path.exists("best.pt"):
            return YOLO("best.pt")

        return YOLO("yolov8n.pt")

    except Exception:
        return None


def extrair_candidatos_etiqueta(imagem_bytes):
    """
    Recebe os bytes de uma imagem e retorna uma lista
    de possíveis recortes de etiquetas.

    Cada candidato possui:
        - imagem: imagem RGB recortada
        - confianca: confiança da detecção
    """

    # ---------------------------------------------------------
    # 1. CONVERTER BYTES PARA IMAGEM
    # ---------------------------------------------------------

    array_imagem = np.frombuffer(imagem_bytes, np.uint8)
    img_bgr = cv2.imdecode(array_imagem, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return []

    altura_original, largura_original = img_bgr.shape[:2]

    candidatos = []

    # ---------------------------------------------------------
    # 2. TENTATIVA COM YOLO
    # ---------------------------------------------------------

    modelo = carregar_modelo_yolo()

    if modelo is not None:

        try:
            resultados = modelo(
                img_bgr,
                conf=0.25,
                iou=0.5,
                verbose=False
            )

            deteccoes = []

            for resultado in resultados:

                if resultado.boxes is None:
                    continue

                for box in resultado.boxes:

                    try:
                        conf = float(box.conf[0])

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].tolist()
                        )

                        deteccoes.append(
                            (conf, x1, y1, x2, y2)
                        )

                    except Exception:
                        continue

            # -------------------------------------------------
            # 3. ORDENAR PELA MAIOR CONFIANÇA
            # -------------------------------------------------

            deteccoes.sort(
                key=lambda item: item[0],
                reverse=True
            )

            # -------------------------------------------------
            # 4. CRIAR OS RECORTES
            # -------------------------------------------------

            for conf, x1, y1, x2, y2 in deteccoes:

                largura_caixa = x2 - x1
                altura_caixa = y2 - y1

                # 4% de margem
                margem_x = int(largura_caixa * 0.04)
                margem_y = int(altura_caixa * 0.04)

                x1_final = max(0, x1 - margem_x)
                y1_final = max(0, y1 - margem_y)

                x2_final = min(
                    largura_original,
                    x2 + margem_x
                )

                y2_final = min(
                    altura_original,
                    y2 + margem_y
                )

                crop_bgr = img_bgr[
                    y1_final:y2_final,
                    x1_final:x2_final
                ]

                if crop_bgr.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(
                    crop_bgr,
                    cv2.COLOR_BGR2RGB
                )

                candidatos.append({
                    "imagem": crop_rgb,
                    "confianca": conf
                })

        except Exception:
            pass

    # ---------------------------------------------------------
    # 5. FALLBACK OPENCV
    # ---------------------------------------------------------

    if not candidatos:

        try:
            gray = cv2.cvtColor(
                img_bgr,
                cv2.COLOR_BGR2GRAY
            )

            # CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            gray = clahe.apply(gray)

            # Gradiente morfológico
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (25, 7)
            )

            gradient = cv2.morphologyEx(
                gray,
                cv2.MORPH_GRADIENT,
                kernel
            )

            # Threshold Otsu
            _, threshold = cv2.threshold(
                gradient,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Fechamento morfológico
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (25, 7)
            )

            threshold = cv2.morphologyEx(
                threshold,
                cv2.MORPH_CLOSE,
                kernel_close
            )

            # Encontrar contornos
            contornos, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            area_original = (
                largura_original * altura_original
            )

            for contorno in contornos:

                x, y, w, h = cv2.boundingRect(contorno)

                if h <= 0:
                    continue

                proporcao = w / h

                area = w * h

                proporcao_area = area / area_original

                # Filtros originais
                if proporcao < 1.5 or proporcao > 6.0:
                    continue

                if proporcao_area < 0.008:
                    continue

                if proporcao_area > 0.70:
                    continue

                # Margem de 15 pixels
                x1 = max(0, x - 15)
                y1 = max(0, y - 15)

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

                candidatos.append({
                    "imagem": crop_rgb,
                    "confianca": 0.5
                })

        except Exception:
            pass

    # ---------------------------------------------------------
    # 6. ORDENAR RESULTADOS
    # ---------------------------------------------------------

    candidatos.sort(
        key=lambda item: item["confianca"],
        reverse=True
    )

    return candidatos
