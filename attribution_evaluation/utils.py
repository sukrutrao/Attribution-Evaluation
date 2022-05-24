def get_augmentation_range(shape, scale, index=0):
    """
    Returns row and column bounds for a grid cell in an N x N grid

    :param shape: Shape of the grid, of the form (B, C, N*H, N*W)
    :type shape: tuple
    :param scale: Grid dimension N
    :type scale: int
    :param index: Index of the grid cell for which to return bounds, defaults to 0
    :type index: int, optional
    :return: y-coordinate, x-coordinate, height H, and width W for the grid cell at the index
    :rtype: tuple
    """
    assert 0 <= index and index < scale * scale
    y_pos = index // scale
    x_pos = index % scale
    h = shape[2] // scale
    w = shape[3] // scale
    y = h * y_pos
    x = w * x_pos
    return y, x, h, w
