def tile(x, dim, num_tile):
    shape = x.shape
    repeat_dim = [1]*(len(shape)+1)
    repeat_dim[dim] = num_tile
    return x.unsqueeze(dim).repeat(*repeat_dim)