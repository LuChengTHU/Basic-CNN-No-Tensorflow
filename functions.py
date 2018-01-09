import numpy as np


def im2col_filter(filter_w):
    '''
    filter_w: shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
    output: shape = c_out x (k * k * c_in)
    '''
    c_out, c_in, kh, kw = filter_w.shape
    return filter_w.reshape(c_out, c_in * kh * kw)


def im2col_input(input_padded, kh, kw, stride=1):
    '''
    input: shape = n (#sample) x c_in (#input channel) x h_padded (#height_padded) x w_padded (#width_padded)
    output: shape = (c_in * kh * kw) x (n * h_out * w_out),
            h_out = (h_padded - kh) / stride + 1
            w_out = (w_padded - kw) / stride + 1

    Here, I referred this website: https://stackoverflow.com/questions/34254679/how-can-i-implement-deconvolution-layer-for-a-cnn-in-numpy/46562181#46562181

    '''
    n, c_in, h_padded, w_padded = input_padded.shape
    h_out = (h_padded - kh) / stride + 1
    w_out = (w_padded - kw) / stride + 1
    col_shape = (c_in, kh, kw, n, h_out, w_out)
    strides = (h_padded * w_padded, w_padded, 1, c_in * h_padded * w_padded, w_padded * stride, stride)
    strides = input_padded.itemsize * np.array(strides)
    output = np.lib.stride_tricks.as_strided(input_padded, shape=col_shape, strides=strides)
    return output.reshape(c_in * kw * kh, n * h_out * w_out)


def conv(input, W):
    '''
    input has been padded.
    '''
    col_w = im2col_filter(W)
    col_input = im2col_input(input, W.shape[2], W.shape[3])
    col_output = np.dot(col_w, col_input)
    h_out = input.shape[2] - W.shape[2] + 1
    w_out = input.shape[3] - W.shape[3] + 1
    output = col_output.reshape(W.shape[0], input.shape[0], h_out, w_out)
    output = np.transpose(output, (1, 0, 2, 3))
    return output


def conv_bias(input, W, b):
    '''
    input has been padded.
    '''
    col_w = im2col_filter(W)
    col_input = im2col_input(input, W.shape[2], W.shape[3])
    col_output = np.dot(col_w, col_input) + b.reshape(-1, 1)
    h_out = input.shape[2] - W.shape[2] + 1
    w_out = input.shape[3] - W.shape[3] + 1
    output = col_output.reshape(W.shape[0], input.shape[0], h_out, w_out)
    output = np.transpose(output, (1, 0, 2, 3))
    return output


def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
            h_out = h_in + 2 * pad - kernel_size + 1
            w_out = w_in + 2 * pad - kernel_size + 1
    '''

    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    return conv_bias(input_padded, W, b)


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''

    grad_output_padded = np.pad(grad_output, ((0, 0), (0, 0), (kernel_size - 1, kernel_size - 1), (kernel_size - 1, kernel_size - 1)), 'constant')
    w_local_grad = np.rot90(np.transpose(W, (1, 0, 2, 3)), 2, axes=(2, 3))
    grad_input = conv(grad_output_padded, w_local_grad)
    if pad != 0:
        grad_input = grad_input[:, :, pad:-pad, pad:-pad]

    # trans_input_padded shape: c_in x n x h_in+2pad x w_in+2pad => n , cin, hin, win
    # trans_grad_output = c_out x n x h_out x w_out => cout x cin x kh x kw
    # out: n, cout, hin-kh+1 ... = c_in, c_out, h_in+2*pad-h_out+1 = kernel, ...
    # grad_w : c_out, c_in, kernel, kernel
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    grad_w = conv(np.transpose(input_padded, (1, 0, 2, 3)), np.transpose(grad_output, (1, 0, 2, 3)))
    grad_w = np.transpose(grad_w, (1, 0, 2, 3))

    grad_b = np.sum(grad_output, axis=(0, 2, 3))

    return grad_input, grad_w, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    n, c_in, h_in, w_in = input.shape
    input_padded = np.pad(input, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    h_out = (h_in + 2 * pad) / kernel_size
    w_out = (w_in + 2 * pad) / kernel_size

    input_padded = input_padded.reshape([n * c_in, 1, h_in + 2*pad, w_in + 2*pad])
    col_input = im2col_input(input_padded, kernel_size, kernel_size, stride=kernel_size).reshape(kernel_size*kernel_size, -1)
    output = np.mean(col_input, axis=0).reshape(n, c_in, h_out, w_out)
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad_input = np.kron(grad_output, np.ones((kernel_size, kernel_size))) / (kernel_size * kernel_size)
    if pad == 0:
        return grad_input
    else:
        return grad_input[:, :, pad:-pad, pad:-pad]
