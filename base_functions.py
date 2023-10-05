import numpy as np
import matplotlib.pyplot as plt
import math
import sympy
def matriz_produto_vetorial(w):
    """
    w CHAPEUZINHO
    vetor coluna 3 elementos
    ex: w = np.array([[1],
                      [2],
                      [3]])
    """
    w_cross = [[ 0      ,-w[2][0], w[1][0]],
               [ w[2][0], 0      ,-w[0][0]],
               [-w[1][0], w[0][0], 0      ]]
    return np.array(w_cross)

def Rodrigues(w,ang): # Igual a R_from_w_theta
    """
    Implementação da Formula de Rodrigues 
    para achar uma matriz de rotacao 
    de um angulo ang em um eixo w
    """

    w_cross = matriz_produto_vetorial(w)
    return np.eye(3) + np.sin(ang)*w_cross + (1-np.cos(ang))*(w_cross @ w_cross)

# criar matrizes de rotação elementares
# ang: ângulo em rad
def Rotacao_x(ang):
    return np.array([
        [1, 0, 0],
        [0, np.cos(ang), -np.sin(ang)],
        [0, np.sin(ang), np.cos(ang)]
    ])

def Rotacao_y(ang):
    return np.array([
        [np.cos(ang), 0, np.sin(ang)],
        [0, 1, 0],
        [-np.sin(ang), 0, np.cos(ang)]
    ])
    
def Rotacao_z(ang):
    return np.array([
        [np.cos(ang), -np.sin(ang), 0],
        [np.sin(ang), np.cos(ang), 0],
        [0,  0, 1]
    ])

def Translacao(r):
    """
    # translacao de um vetor w por r:
    w_aux = translacao(r) @ (np.append(w,[4]).reshape(4,1))
    w_final = w_aux[0:3] #(nao pega o ultimo elemento)
    """
    return np.array([
        [1,0,0,r[0]],
        [0,1,0,r[1]],
        [0,0,1,r[2]],
        [0,0,0,1]
    ])

def transformacao(rotacao: np.array,pose: np.array):
    """
    rotacao é uma matriz 3x3
    pose é um vetor 3x1

    a função concatena por coluna a rotacao e o pose e 
    depois concatena por linha o resultado e [0,0,0,1]
    """

    # return np.array([[rotacao[0][0], rotacao[0][1], rotacao[0][1], pose[0]],
    #                  [rotacao[0][1], rotacao[1][1], rotacao[1][2], pose[1]],
    #                  [rotacao[2][0], rotacao[2][1], rotacao[2][2], pose[2]],
    #                  [0            , 0            , 0            , 1     ]])
    

    return np.row_stack([np.column_stack([rotacao, pose]), np.array([0,0,0,1])])

def traco(R):
    return sum(R[i][i] for i in range(len(R)))

def theta_rotacao(R):
    return np.arccos((traco(R)-1)/2)

def w_rotacao(R):
    """
    A operacao original retorna w CHAPEUZINHO e para converter elevamos a V
    """
    return ElevadoAV((1/(2*np.sin(theta_rotacao(R))))*(R-R.T))

def ElevadoAV(R):
    """
    Inversa da matriz Produto vetorial
    """
    return np.array([R[2][1], R[0][2], R[1][0]]).reshape(3,1)

def quaternion_por_w_theta(theta, w):
    q0=np.cos(theta/2)
    qv=np.sin(theta/2)*w
    # return [q0] + list(qv[:,0])
    return np.array(q0, qv)

def quaternion_por_R(R):
    q0=((1/2)*(1+traco(R)))
    qv_chapeu=(1/4*q0)*(R-R.T)
    qv = ElevadoAV(qv_chapeu)
    return [q0] + list(qv[:,0])# = [q0, qi, qj, qk]

def ang_euler(R):
    phi = np.arctan2(-R[1][2],-R[0][2])
    theta = np.arctan2(-np.sqrt((-R[0][2])**2+(R[1][2])**2), R[2][2])
    psi = np.arctan2(-R[2][1], -R[2][0])
    return phi, theta, psi

def rot_from_a_to_b(a,b):
    xa, ya, za= a[0], a[1], a[2]
    xb, yb, zb= b[0], b[1], b[2]
    return np.array([[xa-xb, xa-yb, xa-zb],
                     [ya-xb, ya-yb, ya-zb],
                     [za-xb, za-yb, za-zb]])

def R_from_w_theta(w,theta): # Igual a Rodrigues
    if type(theta) == sympy.Symbol:
        cos= sympy.cos
        sin= sympy.sin
        v_theta = 1-cos(theta)
    else:
        cos = np.cos
        sin= np.sin
        v_theta = 1-cos(theta)

    return [w[0]**2*v_theta+ cos(theta), w[0]*w[1]*v_theta-w[2]*sin(theta), w[0]*w[2]*v_theta + w[1]*sin(theta)], [w[0]*w[1]*v_theta + w[2]*sin(theta), (w[1]**2)*v_theta + cos(theta), w[1]*w[2]*v_theta - w[0]*sin(theta)],[w[0]*w[2]*v_theta - w[1]*sin(theta), w[1]*w[2]*v_theta + w[0]*sin(theta), w[2]*w[2]*v_theta + cos(theta)]

def plot_vector(matrix):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #VECTOR 1
    ax.quiver(0, 0, 0, matrix[0,0], matrix[0,1], matrix[0,2], color='r', arrow_length_ratio=0.1)
    #VECTOR 2
    ax.quiver(0, 0, 0, matrix[1,0], matrix[1,1], matrix[1,2], color='g', arrow_length_ratio=0.1)
    #VECTOR 3
    ax.quiver(0, 0, 0, matrix[2,0], matrix[2,1], matrix[2,2], color='b', arrow_length_ratio=0.1)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Vector Plot')

    plt.show()