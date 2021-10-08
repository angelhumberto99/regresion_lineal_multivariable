import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('dataset_RegresionLinealMultivariable.csv')
    # obtenemos los datos de X y Y
    x = np.array([df['X1'],df['X2']])
    x = x.transpose()
    y = np.array(df['Y'])

    # parametros
    m, n = x.shape
    a = np.zeros((n+1,1))
    
    # parametros de ajuste
    beta = 0.8
    iter_max = 600
    i = 1

    # Normalizamos los datos
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    xNorm = []
    
    for i in range(n):
        xNorm.append((x[:,i] - mu[i])/sigma[i])
    
    ones = np.ones((1,m))
    x = np.concatenate((ones,np.array(xNorm)))
    x = np.transpose(x)
    n = n+1
    
    # predicción
    h = np.zeros((m,1))
    for i in range(m):
        h[i] = np.sum(np.transpose(a)*x[i])

    # función de costo
    J = (1/(2*m))*np.sum((np.transpose(h)-y)**2);
    convergence = []

    # ciclo principal
    while(i < iter_max):
        for j in range(n):
            a[j] = a[j]-beta*((1/m)*np.sum((np.transpose(h)-y)*x[:,j]))
        for k in range(m):
            h[k] = np.sum(np.transpose(a)*x[k])
        J = (1/(2*m))*np.sum((np.transpose(h)-y)**2);
        convergence.append(J)
        i += 1

    print("J: ", J)
    print("a0: ", a[0])
    print("a1: ", a[1])
    print("a2: ", a[2])
    print("Datos de prueba")

if __name__ == "__main__":
    main()