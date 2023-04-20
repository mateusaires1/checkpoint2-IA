import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# Inicializar as variáveis das mãos mais à esquerda e mais à direita
hand_left = None
hand_right = None

# Defina as regras do jogo
rules = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}

# Inicializa o módulo de detecção de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.0)
kernel = np.ones((5, 5), np.uint8)
# Carrega o vídeo
cap = cv2.VideoCapture('pedra-papel-tesoura.mp4')

# Loop para processar cada frame do vídeo
while cap.isOpened():
    # Lê um frame do vídeo
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    cv2.imshow('teste', frame)

    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    hsv = cv2.cvtColor(opening, cv2.COLOR_RGB2HSV)
    low_branco = np.array([0,0,0], dtype=np.uint8)
    upper_branco = np.array([255,255,254], dtype=np.uint8)
    mask = cv2.inRange(hsv, low_branco, upper_branco)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    h,w, _ = frame.shape



    # Processa o frame com o detector de mãos
    hands_results = hands.process(res)
    
    # Desenha os resultados na imagem original
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Desenha os pontos das mãos na imagem original
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verificar se mãos foram detectadas
            if hands_results.multi_hand_landmarks:
        # Iterar sobre as mãos detectadas
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    # Converter pontos da mão para numpy array
                    hand_points = np.array([[point.x*w, point.y*h] for point in hand_landmarks.landmark], dtype=np.float32)

                    # Encontrar ponto de referência para comparar coordenadas x
                    # Nesse exemplo, estamos usando a ponta do dedo indicador (index finger)
                    ref_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x

                    if hand_landmarks == hands_results.multi_hand_landmarks[0]:
                        # Se for a primeira mão detectada, atribuir a mão mais à esquerda
                        hand_left = hand_points
                        ref_left = ref_x
                    else:
                        # Se for a segunda mão detectada, comparar com a mão anterior e atribuir a mão mais à direita
                        if ref_x > ref_left:
                            hand_right = hand_points
                            hand_left = hand_points
                        else:
                            hand_right = hand_left

                    if hand_left is not None:
                        print(f'Coordenadas da mão mais à esquerda: {hand_left}')
                    if hand_right is not None:
                        print(f'Coordenadas da mão mais à direita: {hand_right}')

                    cv2.drawContours(frame, [cv2.convexHull(hand_left.astype(np.int32))], 0, (0, 0, 0), 2)
                    if hand_right is not None:
                        cv2.drawContours(frame, [cv2.convexHull(hand_right.astype(np.int32))], 0, (0, 0, 0), 2)


            # Verifica quantos dedos estão levantados na mão esquerda
            fingers_left = 0
            if hand_left is not None:
                # Polegar
                if hand_left[4][1] < hand_left[3][1]:
                    fingers_left += 1

                # Dedos restantes
                for i in range(1, 5):
                    if hand_left[i][0] < hand_left[i - 1][0]:
                        fingers_left += 1

            # Repete o processo para a mão direita
            # fingers_right = 0
            # if hand_right_landmarks:
            #     # Polegar
            #     if hand_right_landmarks[4][1] < hand_right_landmarks[3][1]:
            #         fingers_right += 1

            #     # Dedos restantes
            #     for i in range(1, 5):
            #         if hand_right_landmarks[i][2] < hand_right_landmarks[i - 1][2]:
            #             fingers_right += 1


            
            # dedos = [8,12,16,20]
            # contador = 0
            # if hand_right is not None:
            #     if hand_right[4][0] < hand_right[3][0]:
            #         contador += 1
            #     for x in dedos:
            #         if hand_right[x][1] < hand_right[x-2][1]:
            #             contador += 1
            cv2.putText(frame,str(fingers_left),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,0,0),5)            


    # Mostra a imagem com os resultados
    cv2.imshow('Hands Detection', frame)
    if cv2.waitKey(10) == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
