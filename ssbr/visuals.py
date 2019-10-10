def make_tiles(stack, width=5):
    """
    Generate a tileset of images from a set of stack of numpy arrays (RGB images).
    """
    height = math.ceil(len(stack) / width)
    shp = stack[0].shape
    tileset = np.zeros((shp[0] * height, shp[1] * width, shp[2]))
    cnt = 0
    for w in range(width):
        for h in range(height):

            if cnt >= len(stack):
                break

            tileset[h * shp[0]:(h + 1) * shp[0], w * shp[1]:(w + 1) *
                    shp[1], :] = stack[cnt]
            cnt += 1
    return tileset