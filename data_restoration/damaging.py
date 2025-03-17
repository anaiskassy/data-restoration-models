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

def damaging_dataset(dataset,percent=5,rand=False):
    dataset_damaged = dataset.copy()
    for i in range(dataset.shape[0]) :
        dataset_damaged[i,:,:,:] = damaging(dataset[i,:,:,:],percent,rand)
    return dataset_damaged


def damaging_opti(X, n_dim=16):
    '''
    en argumements matrice et un pourcentage de détérioration
    retourne l'image avec un carré blanc
    random : position du carré blanc, si False au centre
    '''
    #n_dim est le nb pixels pour carre blanc
    #X est la taille image d'input, en 3 dim (width, height, nb channels)
    ind = int(round((X.shape[0]-n_dim)/2,0))
    X_damaged = X.copy()
    partie = X[ind:ind+n_dim,ind:ind+n_dim,:]

    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            for k in range(3) :
                if i < ind+n_dim and i >= ind and j >= ind and j < ind + n_dim :
                    X_damaged[i,j,k] = 255

    return X_damaged, partie

def damaging_opti_dataset(dataset,n_dim=16):
    dataset_damaged = dataset.copy()
    #dataset_partie = np.ones((n_dim, n_dim), dtype=np.uint8)
    dataset_partie = dataset[:,:n_dim,:n_dim,:].copy()

    for i in range(dataset.shape[0]) :
        dataset_damaged[i,:,:,:] , dataset_partie[i,:,:,:] = damaging_opti(dataset[i,:,:,:],n_dim)

    return dataset_damaged , dataset_partie


def damaging_opti_normalized(X, n_dim=16):
    '''
    en argumements matrice et un pourcentage de détérioration
    retourne l'image avec un carré blanc
    random : position du carré blanc, si False au centre
    '''
    #n_dim est le nb pixels pour carre blanc
    #X est la taille image d'input, en 3 dim (width, height, nb channels)
    ind = int(round((X.shape[0]-n_dim)/2,0))

    X_damaged = (X.copy()/255) * 2 - 1
    X_damaged_noise = (X.copy()/255) * 2 - 1

    partie = X[ind:ind+n_dim,ind:ind+n_dim,:]

    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            for k in range(3) :
                if i < ind+n_dim and i >= ind and j >= ind and j < ind + n_dim :
                    X_damaged[i,j,k] = 1
                    noise = np.random.uniform(-1,1)
                    X_damaged_noise[i,j,k] = noise

    return X_damaged, X_damaged_noise, partie

def damaging_opti_dataset_normalized(dataset,n_dim=16):
    dataset_damaged = dataset.copy()
    dataset_damaged_noise = dataset.copy()
    #dataset_partie = np.ones((n_dim, n_dim), dtype=np.uint8)
    dataset_partie = dataset[:,:n_dim,:n_dim,:].copy()

    for i in range(dataset.shape[0]) :
        dataset_damaged[i,:,:,:] , dataset_damaged_noise[i,:,:,:], dataset_partie[i,:,:,:] = damaging_opti_normalized(dataset[i,:,:,:],n_dim)

    return dataset_damaged, dataset_damaged_noise, dataset_partie


def postprocessing_dataset(X_full,X_part,n_dim=16):
    for element in range(X_full.shape[0]) :
        image = X_full[element]
        ind = int(round((image.shape[0]-n_dim)/2,0))
        partie = X_part[element]
        if np.max(image) > 1 :
            image = image /255
        if np.max(partie) > 1 :
            partie = partie /255
        rebuild_image = image.copy()
        for i in range(image.shape[0]) :
            for j in range(image.shape[1]) :
                for k in range(3) :
                    if i < ind+n_dim and i >= ind and j >= ind and j < ind + n_dim :
                        rebuild_image[i,j,k] = partie[i-ind,j-ind,k]
        if element == 0 :
            rebuild_X = np.expand_dims(rebuild_image,axis=0)
        else :
            rebuild_X = np.vstack([rebuild_X,np.expand_dims(rebuild_image,axis=0)])
    return rebuild_X
