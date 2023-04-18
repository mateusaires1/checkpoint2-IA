import cv2
import numpy as np

hands_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

rules = {"pedra": "tesoura", "tesoura": "papel", "papel": "pedra"}

cap = cv2.VideoCapture(0)

left_roi = [(0, 0), (320, 480)]
right_roi = [(320, 0), (640, 480)]

def detect_hands(frame, roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hands_model.detectMultiScale(gray, 2, 3)
    for (x, y, w, h) in hands:
        if x > roi[0][0] and y > roi[0][1] and x + w < roi[1][0] and y + h < roi[1][1]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return (x, y, w, h)
    return None

def determine_choice(hand_position):
    if hand_position is None:
        return None
    elif hand_position[2] > hand_position[3]:
        return "pedra"
    elif hand_position[3] > hand_position[2]:
        return "papel"
    else:
        return "tesoura"

def determine_winner(left_choice, right_choice):
    if left_choice is None:
        return "Mão direita ganhou!"
    elif right_choice is None:
        return "Mão esquerda ganhou!"
    elif rules[left_choice] == right_choice:
        return "Mão direita ganhou!"
    else:
        return "Mão esquerda ganhou!"

while True:
    # Captura o vídeo da webcam
    ret, frame = cap.read()

    # Detecta as mãos dos jogadores
    left_hand_position = detect_hands(frame, left_roi)
    right_hand_position = detect_hands(frame, right_roi)

    # Determina a escolha de cada jogador com base na posição da mão detectada
    left_choice = determine_choice(left_hand_position)
    right_choice = determine_choice(right_hand_position)

    # Determina o vencedor do jogo com base nas escolhas dos jogadores
    winner = determine_winner(left_choice, right_choice)

    # Exibe o resultado na tela
    cv2.putText(frame, winner, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Left: {}".format(left_choice), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Right: {}".format(right_choice), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pedra, Papel e Tesoura', frame)

    # Sai do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos utilizados
cap.release()
cv2.destroyAllWindows()
