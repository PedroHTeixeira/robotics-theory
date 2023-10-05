import numpy as np
from sympy import *
from base_functions import *

# Questao 1 ------------------------------------------------------------------



# Questao 2 ------------------------------------------------------------------
comment = """
Guia pra plotar: 

      x    y   z
    [3/5, -4/5, 0],
    [0, 0, 1],
    [-4/5, -3/5, 1]

    xb estará em (3/5, 0, -4/5)
    yb estará em (-4/5, 0, -3/5)
    zb estará em (0, 1, 0)

    Logo, eixo_b será a coluna correspondente na matriz de rotação

    Para plotar Eb:

    Eb = Ea @ Rab

"""

# Questao 3 ------------------------------------------------------------------
rot_result = Rotacao_y(np.pi/2)@Rotacao_z(-np.pi/4)
print("rot_result",rot_result)
q3_passo1 = [w_rotacao(rot_result), theta_rotacao(rot_result)]
print("q3_passo1",q3_passo1)
q3_passo2 = quaternion_por_R(rot_result) #q3_passo2
print("q3_passo2",q3_passo2)
phi,theta,psi = ang_euler(rot_result) # q3_passo3
print("q3_passo3",phi,theta,psi)


# Rotação em eixo corrente, multiplica pela direita
# Rotação em eixo inercial, multiplica pela esquerda

# to no zero, fui pro 1, volto pro zero aplico a rot e vou pro 1

# Questao 4 ------------------------------------------------------------------
# Passo 1: Translação de 3 no eixo x
q4_passo1 = Translacao([3,0,0])
print("q4_passo1", q4_passo1)
plot_vector(q4_passo1)
# Passo 2: Rotacao de pi/2 no eixo corrente z
q4_passo2 = q4_passo1 @ transformacao(Rotacao_z(np.pi/2), [0,0,0])
print("q4_passo2", q4_passo2)
# Passo 3: Translacao de 1 no eixo inercial y 
q4_passo3 = Translacao([0,1,0])@q4_passo1
print("q4_passo3", q4_passo3)
# Passo 4: Rotacao de -pi/4 no inercial x
q4_passo4 = transformacao(Rotacao_z(-np.pi/4), [0,0,0])@q4_passo3
print("q4_passo4", q4_passo4)

#Questao 5 ------------------------------------------------------------------
theta = symbols('theta')
R1= R_from_w_theta([0,0,1], theta)
print(R1)
R2= R_from_w_theta([-1/(3)**(1/2),-1/(3)**(1/2),-1/(3)**(1/2)], theta)
print(R2)
print(exp(np.array(R1)@np.array(R2)))