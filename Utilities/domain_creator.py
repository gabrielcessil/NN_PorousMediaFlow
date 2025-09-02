import numpy as np
from scipy.ndimage import gaussian_filter
import time
import random
import networkx as nx
import os

def save_as_raw(array, folder_name, simulation_name):
    """
    Saves a 2D numpy array as a .raw file in 8-bit integer format.

    Parameters:
    - array (numpy.ndarray): The 2D array to be saved.
    - filename (str): The name of the file (including .raw extension).

    Returns:
    - None
    """
    # Ensure the array is 2D
    if len(array.shape) != 2:
        raise ValueError("Only 2D arrays are supported.")
    
    # Convert the array to 8-bit integer format
    array_8bit = array.astype(np.uint8)   
    
    filename = os.path.join(folder_name, simulation_name + ".raw")
    array_8bit.tofile(filename)
    print(f"Volume salvo como {filename}")
    
def reflect_domain(domain):
    return np.hstack((domain, np.fliplr(domain)))
def get_reflected_shape(shape): return (shape[0], shape[1], 2*shape[2])

def create_media_fromPorosity(shape, minor_correlation_length, porosity):
    """
    Gera um meio poroso artificial utilizando campos correlacionados.

    Parâmetros:
        shape (tuple): Tamanho da imagem gerada (altura, largura).
        minor_correlation_length (float): Comprimento de correlação (desfoque Gaussiano).
        threshold (float): Limite para segmentação (0 a 1).
        bias (float): Probabilidade de as células iniciais serem sólidas (0 a 1).

    Retorno:
        numpy.ndarray: Imagem binária representando a estrutura porosa.
    """
    # Passo 1: Geração de números aleatórios uniformemente distribuídos
    random_field = np.random.rand(*shape)
    random_field = (random_field < porosity).astype(np.float64)  # Converte para float
    
    # Passo 2: Aplicação de filtro Gaussiano para criar correlação
    correlated_field = gaussian_filter(random_field, sigma=minor_correlation_length/2)

    # Passo 3: Normalização para intervalo [0, 1]
    correlated_field = (correlated_field - correlated_field.min()) / (correlated_field.max() - correlated_field.min())
    
    # Passo 4: Segmentação binária (0 = sólido, 1 = poro)
    threshold = porosity
    ajuste = (porosity-np.mean(correlated_field))/2
    binary_field = (correlated_field < threshold-ajuste).astype(np.uint8)
    
    return binary_field

def create_media_fromTortuosity(shape, n_paths, path_width=1, minor_correlation_length=2, tortuosity=1):
    """
    Gera uma mídia porosa 2D com caminhos aleatórios conectados e filtro gaussiano aplicado.

    Parameters:
        shape (tuple): Tamanho do array 2D (altura, largura).
        n_paths (int): Número de caminhos aleatórios a serem gerados.
        path_width (int): Largura dos caminhos (número de células com valor 1).
        noise_level (float): Nível de ruído a ser adicionado (0 a 1).
        minor_correlation_length (float): Desvio padrão do filtro gaussiano.
        tortuosity (int): Controle da tortuosidade dos caminhos.

    Returns:
        numpy.ndarray: Array 2D representando a mídia porosa.
    """
    height, width = shape
    media = np.zeros(shape, dtype=int)

    for i_path in range(n_paths):
        # Define uma posição inicial na interface esquerda
        random.seed((int(time.time() * 1000) % 2**32)*(i_path+1))
        y = random.randint(0, height - 1)
        x = 0 #if i_path==0 else random.randint(0, width - 1)

        # Caminho aleatório para a interface direita
        while x < width-1:
                        
            # Probabilidades para ir na direção correta ou desviar
            prob_reto = int(100 // (tortuosity*2))  # Proporção de passos "reta"
            prob_desvio = int((100 - prob_reto))  # Proporção de desvios em y ou direção oposta
            
            # 3 times is the minimum step to count as 1 path_width, because of the corners "shortcuts" of succesive curves
            x_step = int(path_width*2)
            y_step = int(path_width*2*(tortuosity*2)) # Quanto maior a tortuosidade maior devem ser os passos grandes pras laterais
            
            # Direção do passo em x e y
            y_dir = random.choice([0] * prob_reto + [1] * int(prob_desvio/4) + [-1] * int(prob_desvio/4))
            x_dir = random.choice([1] * prob_reto) #+ [-1]* int(prob_desvio/4))
            
            # Calcula o passo
            next_x = x + x_dir*x_step
            next_y = y + y_dir*y_step
            
            # Ajusta para que o próximo ponto fique dentro dos limites
            next_y = max(0, min(height - 1, next_y))
            next_x = min(width - 1, next_x)

            # Desenha uma linha entre o ponto atual e o próximo
            draw_line_with_thickness(media, x, y, next_x, next_y, path_width)
            
            # Atualiza o ponto atual
            x, y = next_x, next_y

    # Aplica o filtro gaussiano e binariza novamente
    porous_media = gaussian_filter(media.astype(float), sigma=minor_correlation_length)
    
    
    porous_media = (porous_media > 0.5).astype(int)

    return porous_media


def create_media_fromTortuosity_and_porosity(shape, path_width, minor_correlation_length, tortuosity, porosity):

    n_paths = int(np.ceil(porosity*shape[0]/path_width)/2)
    midia_tubos = create_media_fromTortuosity(shape, n_paths, path_width, minor_correlation_length, tortuosity)

    return midia_tubos



def create_media_fromPorousTortuosity_and_porosity(shape, path_width=1, minor_correlation_length=2, tortuosity=1, porosity=0.5):
    """
    Gera uma mídia porosa sintética com grãos organizados ao redor dos caminhos gerados por create_media_fromTortuosity.
    
    Parâmetros:
    - shape: tuple, formato da matriz gerada (ex: (100, 100)).
    - n_paths: int, número de caminhos a serem criados.
    - path_width: int, largura dos caminhos na mídia.
    - minor_correlation_length: float, controle da suavização (filtro gaussiano).
    - tortuosity: float, nível de tortuosidade dos caminhos.
    - porosity: float, nível geral de porosidade (0.0 a 1.0).
    
    Retorna:
    - binary_field: np.ndarray, matriz binária representando a mídia (1 = grão, 0 = poro).
    """
    
    n_paths = int(np.ceil(porosity*shape[0]/path_width))
    midia_tubos = create_media_fromTortuosity(shape, n_paths, path_width, minor_correlation_length, tortuosity)
    
    deviation = path_width//2
    noise = np.random.normal(loc=0, scale=deviation, size=midia_tubos.shape)

    # Add noise to the original array
    modified_array = midia_tubos + noise

    
    # Etapa 4: Aplicação de filtro Gaussiano para criar granularidade
    smoothed_field = gaussian_filter(modified_array, sigma=minor_correlation_length/1.25)
    # Etapa 5: Normalização para intervalo [0, 1]
    smoothed_field = (smoothed_field - smoothed_field.min()) / (smoothed_field.max() - smoothed_field.min())

    # Passo 4: Segmentação binária (0 = sólido, 1 = poro)
    binary_field = (smoothed_field > 0.5).astype(np.uint8)
    
    return binary_field,midia_tubos



def draw_line_with_thickness(array, x0, y0, x1, y1, thickness):
    """
    Desenha uma linha com espessura no array entre os pontos (x0, y0) e (x1, y1).

    Parâmetros:
        array (numpy.ndarray): Array 2D no qual a linha será desenhada.
        x0, y0 (int): Coordenadas do ponto inicial.
        x1, y1 (int): Coordenadas do ponto final.
        thickness (int): Espessura da linha (número de células preenchidas ao redor do caminho).
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        for i in range(-thickness // 2, thickness // 2 + 1):
            for j in range(-thickness // 2, thickness // 2 + 1):
                nx, ny = x0 + i, y0 + j
                if 0 <= nx < array.shape[1] and 0 <= ny < array.shape[0]:
                    array[ny, nx] = 1
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
def calcula_tortuosidade(matriz, num_pontos_iniciais=10):
    """
    Calcula a tortuosidade do caminho para 10 células aleatórias na face esquerda da matriz
    que possuem valor 1 e retorna a tortuosidade média.
    
    Parâmetros:
    - matriz: numpy array binário (0's e 1's), onde 1 representa o caminho percorrido.
    - num_pontos_iniciais: Número de células aleatórias na face esquerda para iniciar a busca (default: 10).
    
    Retorna:
    - Tortuosidade média: média das razões entre o comprimento do menor caminho e o caminho direto.
    """
    # Verificar dimensões da matriz
    altura, largura = matriz.shape
    
    # Criar um grafo a partir da matriz
    grafo = nx.grid_2d_graph(altura, largura)  # Cria um grafo de grade com todas as conexões possíveis
    # Remover arestas que correspondem a células com valor 0
    for i in range(altura):
        for j in range(largura):
            if matriz[i, j] == 0:
                grafo.remove_node((i, j))
    
    # Encontrar os pontos na face esquerda com valor 1
    pontos_esquerda = [(i, 0) for i in range(altura) if matriz[i, 0] == 1]
    pontos_direita = [(i, largura-1) for i in range(altura) if matriz[i, largura-1] == 1]
    
    num_pontos_iniciais = min(len(pontos_esquerda), num_pontos_iniciais)

    # Selecionar num_pontos_iniciais células aleatórias da face esquerda
    pontos_iniciais = random.sample(pontos_esquerda, num_pontos_iniciais)

    tortuosidades = []
    
    for ponto_inicial in pontos_iniciais:
        #menor_tortuosidade = float('inf')  # Inicializa com valor muito alto
        ponto_final = pontos_direita[0]
        try:
            
            # Definir a heurística (distância Euclidiana ate a interface direita)
            def euclidean_heuristic(source, target):
                return np.sqrt((source[0] - target[0])**2)
            
            # Calcular o menor caminho usando o algoritmo A*
            path = nx.astar_path(
                grafo, 
                source=ponto_inicial, 
                target=ponto_final, 
                heuristic=euclidean_heuristic, 
                weight="weight"
            )
            
            # Comprimento do menor caminho
            menor_caminho = sum(
                np.sqrt((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2)
                for i in range(len(path) - 1)
            )
            
            tortuosidade = menor_caminho / largura
        
        except nx.NetworkXNoPath:
            continue
        
        tortuosidades.append(tortuosidade)
    
    # Calcular a tortuosidade média
    tortuosidade_media = np.mean(tortuosidades)
    tortuosidade_std = np.std(tortuosidades)
    return tortuosidade_media, tortuosidade_std

def calcula_porosidade(matriz, fluid_default_value=1):
    """
    Calcula o número de pontos iguais a 1 em uma matriz binária.
    
    Parâmetros:
    - matriz: numpy array binário (0's e 1's).
    
    Retorna:
    - Número de pontos iguais a 1 na matriz.
    """
    # Contar o número de elementos iguais a 1
    return np.sum(matriz == fluid_default_value)/matriz.size


# FINALIZAR NOVOS METODOS ABAIXO


