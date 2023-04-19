import cv2
import mediapipe as mp
import numpy as np

def identify_gesture(hand_landmarks):
    """
    Identifica qual jogada foi feita com base nas posições dos dedos da mão.
    Retorna uma string representando a jogada identificada.
    """
    # Define as posições dos pontos que representam cada dedo
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # Calcula as distâncias entre os dedos da mão
    distances = calculate_distances([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])

    # Identifica qual dedo está mais próximo do polegar (indicador para papel, resto da mão para pedra)
    if distances[0] == min(distances):
        return "Papel"
    else:
        # Identifica a posição do dedo médio em relação aos dedos anelar e mindinho (maior distância para tesoura)
        if distances[3] > distances[2]:
            return "Tesoura"
        else:
            return "Pedra"

# Função que calcula as distâncias entre os pontos das mãos
def calculate_distances(hand_positions):
    distances = []
    for i in range(20):
        for j in range(i+1, 21):
            distance = np.linalg.norm(hand_positions[i] - hand_positions[j])
            distances.append(distance)
    return distances

# Função que identifica a jogada a partir das distâncias entre os pontos
def identify_move(distances):
    # Pedra
    if np.all(np.array(distances) < 0.025):
        return 'Pedra'
    # Tesoura
    elif np.all(np.array(distances) > 0.04):
        return 'Tesoura'
    # Papel
    else:
        return 'Papel'

# Função que compara as jogadas dos jogadores e determina o vencedor
def compare_moves(move1, move2):
    if move1 == move2:
        return 'Empate'
    elif move1 == 'Pedra':
        if move2 == 'Tesoura':
            return 'Jogador 1 Venceu'
        else:
            return 'Jogador 2 Venceu'
    elif move1 == 'Papel':
        if move2 == 'Pedra':
            return 'Jogador 1 Venceu'
        else:
            return 'Jogador 2 Venceu'
    elif move1 == 'Tesoura':
        if move2 == 'Papel':
            return 'Jogador 1 Venceu'
        else:
            return 'Jogador 2 Venceu'

# Loop para processar cada quadro do vídeo
while True:
    # Inicializa o objeto Hands
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Carrega o vídeo e inicializa a variável para armazenar o resultado das jogadas
    cap = cv2.VideoCapture('pedra-papel-tesoura.mp4')
    result = ""
    # Lê o próximo quadro do vídeo
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Detecta as mãos no video
    hands_results = hands.process(rgb)
    hands_landmarks = hands_results.multi_hand_landmarks

    # Adiciona linhas nas maos
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        results = hands.process(frame)

    # Se duas mãos foram detectadas, calcula as distâncias entre os dedos e identifica as jogadas dos jogadores
    if hands_landmarks and len(hands_landmarks) == 2:
        # Extrai as coordenadas dos pontos das mãos detectadas
        hand_positions = []
        for hand_landmarks in hands_landmarks:
            landmarks_list = np.zeros((21, 2))
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks_list[i] = [landmark.x, landmark.y]
            hand_positions.append(landmarks_list)

        # Calcula as distâncias entre os dedos
        distances = calculate_distances(hand_positions)

        # Identifica qual jogada cada jogador fez
        player1_play = identify_gesture(hand_positions[0])
        player2_play = identify_gesture(hand_positions[1])

# Determina o resultado da rodada
        if player1_play == player2_play:
            result = "Empate"
        elif player1_play == "Pedra":
            if player2_play == "Tesoura":
                result = "Jogador 1 venceu"
            else:
                result = "Jogador 2 venceu"
        elif player1_play == "Papel":
            if player2_play == "Pedra":
                result = "Jogador 1 venceu"
            else:
                result = "Jogador 2 venceu"
        elif player1_play == "Tesoura":
            if player2_play == "Papel":
                result = "Jogador 1 venceu"
            else:
                result = "Jogador 2 venceu"

        # Desenha o resultado da rodada na imagem
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Mostra a imagem com o resultado da rodada
        cv2.imshow("Jogo Pedra, Papel e Tesoura", frame)

        # Aguarda o pressionamento da tecla 'q' para encerrar o programa
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

# Libera os recursos utilizados
cap.release()
cv2.destroyAllWindows()