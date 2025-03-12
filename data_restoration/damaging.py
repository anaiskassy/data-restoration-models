import numpy as np

def damaging(X : np.ndarray ,percent : int,random : bool = True ):
    '''
    en argumements matrice et un pourcentage de détérioration
    retourne l'image avec un carré blanc
    random : position du carré blanc, si False au centre
    '''
    taille = round(np.sqrt(percent/100*X.shape[0]**2),0).astype(int)
    if random :
        ind_x = np.random.randint(0,X.shape[0]-taille)
        ind_y = np.random.randint(0,X.shape[0]-taille)
    else :
        ind_x = round((X.shape[0]-taille)/2,0)
        ind_y = ind_x

    X_damaged = X.copy()

    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            for k in range(3) :
                if i < ind_x+taille and i >= ind_x and j >= ind_y and j < ind_y + taille :
                    X_damaged[i,j,k] = 255

    return X_damaged
