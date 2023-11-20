
def pad_latent(latent_space_shape, n_downsamplings):
    '''
    Finds the new latent space shape making possible to train the unet
    '''

    divisor = 2**n_downsamplings
    adjust = False
    for i in latent_space_shape[1:]:
        if i%divisor != 0:
            adjust = True
            break
    if adjust:
        new_shape = []
        for i in latent_space_shape[1:]:
            floating_i = i
            ok = False
            while not ok:
                if floating_i%divisor != 0:
                    floating_i += 1

                else:
                    new_shape.append(floating_i)
                    ok = True
        return True, [latent_space_shape[0]] + new_shape
    else:
        return False, list(latent_space_shape)
