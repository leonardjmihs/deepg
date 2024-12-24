import argparse
import csv
import PIL
import os
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation, PillowWriter
# sys.path.insert(0, '../ELINA/python_interface/')
import time

EPS = 10**(-9)
n_rows, n_cols, n_channels = 0, 0, 0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize(image, means, stds, dataset, is_conv):
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    else:
        for i in range(3072):
            image[i] = (image[i] - means[i % 3]) / stds[i % 3]


def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def show_ascii_spec(lb, ub):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')

def compute_lin_image(frame, bound_tup, num_frames=1):
    lb_weights = bound_tup[0]
    ub_weights = bound_tup[1]
    lb_biases = bound_tup[2]
    ub_biases = bound_tup[3]
    LB = bound_tup[4]
    UB = bound_tup[5]

    i_size = lb_weights.shape[0]
    j_size = lb_weights.shape[1]
    n_chan = lb_weights.shape[2]

    image_lb = np.zeros((i_size, j_size, n_chan))
    image_ub = np.zeros((i_size, j_size, n_chan))
    param = np.linspace(LB, UB, num_frames)[int(frame)]

    for i in range(i_size):
        for j in range(j_size):
            for k in range(n_chan):
                image_lb[i,j,k] = lb_weights[i,j,k] * param + lb_biases[i, j,k]
                image_ub[i,j,k] = ub_weights[i,j,k] * param + ub_biases[i, j,k]
                # image_lb[i,j,k] = lb_biases[i,j,k] * param + lb_weights[i, j,k]
                # image_ub[i,j,k] = ub_biases[i,j,k] * param + ub_weights[i, j,k]
    return image_lb, image_ub

def view_linfun(frame, bound_tup, x_original, num_frames=1):
    # Create a figure and axis as before
    fig, axs = plt.subplots(1, 5, figsize=(50,10))
    # Create an animation writer
    writer = PillowWriter(fps=10)
    # safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator, _, LB, UB = bound_tup
    lb_weights = bound_tup[0]
    ub_weights = bound_tup[1]
    lb_biases = bound_tup[2]
    ub_biases = bound_tup[3]
    LB = bound_tup[4]
    UB = bound_tup[5]
    params = np.linspace(LB, UB, num_frames)
    # Start recording the animation
    with writer.saving(fig, "rotation_animation.gif", 100):
        i = 0
        cb0 = None
        cb1 = None

        lower_fun, upper_fun = compute_lin_image(LB, bound_tup, num_frames)

        # Initial bounds images
        # im = axs[3].imshow(bounds.lower.reshape(x_original.shape))
        im = axs[3].imshow(lower_fun)
        axs[3].set_title(f"lower")
        plt.colorbar(im, ax=axs[3])
        im = axs[4].imshow(upper_fun)
        axs[4].set_title(f"upper")
        plt.colorbar(im, ax=axs[4])
        
        for frame in range(num_frames):
            # idx = int(frame/num_frames*image_samples.shape[0])
            # image = image_samples[idx,:,:,:,0]
            # print(param)
            lower_fun, upper_fun = compute_lin_image(frame, bound_tup, num_frames)
            # print(np.max(lower_fun))
            if cb0 is not None:
                cb0.remove()
            axs[0].clear()
            im = axs[0].imshow(lower_fun)
            cb0 = plt.colorbar(im, ax=axs[0])
            axs[0].set_title(f"Lower for ({params[frame].item()}) degrees rotation {i}")
            if cb1 is not None:
                cb1.remove()
            axs[1].clear()
            im = axs[1].imshow(upper_fun)
            cb1 = plt.colorbar(im, ax=axs[1])
            axs[1].set_title(f"Upper for ({params[frame].item()}) degrees rotation {i}")

            axs[2].clear()
            image = rotate_full(x_original, params[frame].item())
            # image = 
            axs[2].imshow(image)
            axs[2].set_title(f"Original")
            writer.grab_frame()  # Capture each frame for the animation

            # im = axs[0].imshow(lower_fun(theta).reshape(x_original.shape))
            # cb0 = plt.colorbar(im, ax=axs[0])
            # axs[0].set_title(f"Lower for ({round(180 / jnp.pi * theta)}) degrees rotation {i}")
            # axs[1].clear()
            # if cb1 is not None:
            #     cb1.remove()
            # im = axs[1].imshow(upper_fun(theta).reshape(x_original.shape))
            # cb1 = plt.colorbar(im, ax=axs[1])
            # axs[1].set_title(f"Upper for ({round(180 / jnp.pi * theta)}) degrees rotation {i}")
            # axs[2].clear() 
            # axs[2].imshow(rotate_full(ks, ls, x_original, theta).reshape(x_original.shape))
            # axs[2].set_title(f"Original")
            # writer.grab_frame()  # Capture each frame for the animation
            i += 1
    plt.show()
        
def vis_pixel_curves(i, j, bound_tup, x_original):
    # safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator, _, LB, UB = bound_tup
    lb_weights = bound_tup[0]
    ub_weights = bound_tup[1]
    lb_biases = bound_tup[2]
    ub_biases = bound_tup[3]
    LB = bound_tup[4]
    UB = bound_tup[5]

    num_frames = 40
    fig, axs = plt.subplots(1, 1, figsize=(50,10))
    lbs = []    
    ubs = []
    pix = []
    params = np.linspace(LB, UB, num_frames)

    for param in params[::-1]:
        im = rotate_full(x_original, param)
        pix.append(im[i,j]) 
    plt.plot(params, pix, "g", label='true value')


    for param in params[::-1]:
        lbs.append(lb_weights[i,j,0] * param + lb_biases[i, j,0])
        ubs.append(ub_weights[i,j,0] * param + ub_biases[i, j,0])

    plt.plot(params[:], lbs, "r", label='lb')
    plt.plot(params[:], ubs, "b", label='ub')
    # plt.ylim([0.4,0.9])
    plt.ylim([-0.1,1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    plt.legend()
    plt.savefig(f"imgs/{i}_{j}.png")
    # plt.show()

def vis_pixel_attack(i, j, bound_tup, x_original, params, attack_lb, attack_ub):
    # safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator, _, LB, UB = bound_tup
    num_frames = 40
    fig, axs = plt.subplots(1, 1, figsize=(50,10))
    lb_weights = bound_tup[0]
    ub_weights = bound_tup[1]
    lb_biases = bound_tup[2]
    ub_biases = bound_tup[3]

    lbs = []    
    ubs = []
    pix = []
    g_truth = []

    for param in params:
        # breakpoint()
        im = rotate_full(x_original, param)
        pix.append(im[i,j]) 


    for it in range(len(params)):
        # lbs.append(attack_lb[i,j,it])
        # ubs.append(attack_ub[i,j,it])
        g_truth.append(attack_lb[i,j,it])
        lbs.append(lb_weights[i,j,0] * params[it] + lb_biases[i, j,0])
        ubs.append(ub_weights[i,j,0] * params[it] + ub_biases[i, j,0])
    
    params = params * 180/np.pi
    plt.plot(params[:], lbs, "ro", label='lb')
    plt.plot(params[:], ubs, "bo", label='ub')
    plt.plot(params[:], g_truth, "ko", label='ub')
    plt.plot(params, pix, "go", label='true value')
    plt.ylim([-0.1,1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.title(f"pixel:  [{i},{j}]")

    plt.legend()
    # plt.show()

def vis_pixel_rotations(bound_tup, x_original, params, attack_lb, attack_ub):
    num_frames=40
    fig, axs = plt.subplots(1, 3, figsize=(50,10))
    writer = PillowWriter(fps=10)
    lb_weights = bound_tup[0]
    ub_weights = bound_tup[1]
    lb_biases = bound_tup[2]
    ub_biases = bound_tup[3]
    a = np.argsort(params, 0)

    # Start recording the animation
    with writer.saving(fig, "rotation_animation.gif", 100):
        i = 0
        cb0 = None
        cb1 = None
        cb2 = None

        f_truth = rotate_full(x_original, params[a[i]][0][0])
        g_truth = attack_lb[:,:,a[i]]

        im = axs[0].imshow(f_truth)
        axs[0].set_title("us")
        plt.colorbar(im, ax=axs[0])
        im = axs[1].imshow(g_truth)
        axs[1].set_title("deepg")
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(f_truth-g_truth[:,:,0])
        plt.colorbar(im, ax=axs[2])
        breakpoint()
        
        for i in range(len(params[:])):
            print(params[a[i]][0][0])
            f_truth = rotate_full(x_original, params[a[i]][0][0])
            g_truth = attack_lb[:,:,a[i]]

            if cb0 is not None:
                cb0.remove()
            axs[0].clear()
            im = axs[0].imshow(f_truth)
            cb0 = plt.colorbar(im, ax=axs[0])
            axs[0].set_title(f"Us for ({params[a[i]][0][0].item()*180/np.pi}) degrees rotation {i}")

            if cb1 is not None:
                cb1.remove()

            axs[1].clear()
            im = axs[1].imshow(g_truth)
            cb1 = plt.colorbar(im, ax=axs[1])
            axs[1].set_title(f"deepg for ({params[a[i]][0][0].item()*180/np.pi}) degrees rotation {i}")

            if cb2 is not None:
                cb2.remove()

            axs[2].clear()
            # breakpoint()
            image = f_truth-g_truth[:,:,0] 
            axs[2].imshow(image)
            # cb2 = plt.colorbar(image, ax=axs[2])
            axs[2].set_title(f"Difference")
            writer.grab_frame()  # Capture each frame for the animation


    # f_truth = []
    # g_truth = []



    # params = params * 180/np.pi

    # plt.plot(params[:], g_truth, "ko", label='deepg')
    # plt.plot(params[:], f_truth, "go", label='us')
    # plt.ylim([-0.1,1.1])
    # # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)

    # plt.legend()
    plt.show()


def rotate_full( x_original, theta):

    # corrects the orientation of rotation
    theta = -theta
    ks_raw = np.arange(x_original.shape[0])

    ls_raw = np.arange(x_original.shape[1])
    xx, yy = np.meshgrid(ls_raw, ks_raw)
    kls = np.dstack([yy, xx]).reshape(-1, 2)
    ks = kls[:, 0] - (x_original.shape[0] - 1) / 2
    ls = kls[:, 1] - (x_original.shape[1] - 1) / 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # defines rotated coordinate grid
    iprimes = ks * cos_theta - ls * sin_theta                  # (N,)   N = np.prod(original.shape)
    jprimes = ls * cos_theta + ks * sin_theta                  # (N,)

    # interpolation between the two grids
    # L1 distance of any row in original grid to any row in rotated grid
    L1_is = np.abs(ks - iprimes[:, np.newaxis])                        # (N, N)
    # L1 distance of any col in original grid to any col in rotated grid
    L1_js = np.abs(ls - jprimes[:, np.newaxis])                        # (N, N)
    
    weight_heights = np.maximum(1 - L1_is, 0)                           # (N, N)
    weight_widths = np.maximum(1 - L1_js, 0)                            # (N, N)

    influence_matrix = (weight_heights * weight_widths)                  # (N, N)

    xijprime = np.dot(                                                  # (N,)
        influence_matrix,
        x_original.reshape(
            -1,
        ),
    )
    xijprime = xijprime.reshape(x_original.shape)
    return xijprime

def main():
    parser = argparse.ArgumentParser(description='Analyze NN.')
    parser.add_argument('--dataset', type=str, default='custom', help='Dataset')
    parser.add_argument('--data_dir', type=str, help='Directory which contains data')
    parser.add_argument('--num_params', type=int, default=1, help='Number of transformation parameters')
    parser.add_argument('--test_idx', type=int, default=None, help='Index to test')
    parser.add_argument('--debug', action='store_true', help='Whether to display debug info')
    parser.add_argument('--attack', action='store_true', help='whether to display attack info')

    args = parser.parse_args()
    global n_rows, n_cols, n_channels
    # n_rows, n_cols, n_channels = 10, 10, 1
    n_rows, n_cols, n_channels = 20, 20, 1
    csvfile = open('datasets/{}_test.csv'.format(args.dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    tot_time = 0


    for i, test in enumerate(tests):
        if args.test_idx is not None and i != args.test_idx:
            continue

        image = np.float64(test[1:len(test)])
        # breakpoint()
        spec_lb = np.copy(image)
        spec_ub = np.copy(image)
        dim = n_rows * n_cols * n_channels
        k = args.num_params + 1 + 1 + dim


        attack_imgs, checked, attack_pass = [], [], 0

        print('tot attacks: ', len(attack_imgs))
        specs_file = os.path.join(args.data_dir, '{}.csv'.format(i))
        attacks_file = os.path.join(args.data_dir, 'attack_{}.csv'.format(i))
        begtime = time.time()
        with open(specs_file, 'r') as fin:
            lines = fin.readlines()
            print('Number of lines: ', len(lines))
            print('K: ', k)
            assert len(lines) % k == 0

            spec_lb = np.zeros(args.num_params + dim)
            spec_ub = np.zeros(args.num_params + dim)

            lexpr_cst, uexpr_cst = [], []
            lexpr_weights, uexpr_weights = [], []
            lexpr_dim, uexpr_dim = [], []
            
            for i, line in enumerate(lines):
                if i % k < args.num_params:
                    # read specs for the parameters
                    values = np.array(list(map(float, line[:-1].split(' '))))
                    assert values.shape[0] == 2
                    param_idx = i % k
                    spec_lb[dim + param_idx] = values[0]
                    spec_ub[dim + param_idx] = values[1]
                    if args.debug:
                        print('parameter %d: [%.4f, %.4f]' % (param_idx, values[0], values[1]))
                elif i % k == args.num_params:
                    # read interval bounds for image pixels
                    values = np.array(list(map(float, line[:-1].split(','))))
                    spec_lb[:dim] = values[::2]
                    spec_ub[:dim] = values[1::2]
                    # if args.debug:
                    #     show_ascii_spec(spec_lb, spec_ub)
                elif i % k < k - 1:
                    # read polyhedra constraints for image pixels
                    tokens = line[:-1].split(' ')
                    assert len(tokens) == 2 + 2*args.num_params + 1

                    bias_lower, weights_lower = float(tokens[0]), list(map(float, tokens[1:1+args.num_params]))
                    assert tokens[args.num_params+1] == '|'
                    bias_upper, weights_upper = float(tokens[args.num_params+2]), list(map(float, tokens[3+args.num_params:]))
                    
                    assert len(weights_lower) == args.num_params
                    assert len(weights_upper) == args.num_params
                    
                    lexpr_cst.append(bias_lower)
                    uexpr_cst.append(bias_upper)
                    for j in range(args.num_params):
                        lexpr_dim.append(dim + j)
                        uexpr_dim.append(dim + j)
                        lexpr_weights.append(weights_lower[j])
                        uexpr_weights.append(weights_upper[j])
                else:
                    assert(line == 'SPEC_FINISHED\n')
                    if args.debug:
                        print('Running the analysis...')
        if args.attack:
            with open(attacks_file, 'r') as fin:
                lines = fin.readlines()
                number_attacks = int(len(lines)/(args.num_params+1))
                params_list = np.zeros((number_attacks, args.num_params))
                attack_lb_list = np.zeros((n_rows,n_cols, number_attacks))
                attack_ub_list = np.zeros((n_rows,n_cols, number_attacks))
                for j in range(0, len(lines), args.num_params+1):
                    params = [float(line[:-1]) for line in lines[j:j+args.num_params]]
                    tokens = lines[j+args.num_params].split(',')
                    values = np.array(list(map(float, tokens)))

                    attack_lb = values[::2]
                    attack_ub = values[1::2]
                    
                    index =int(j/(args.num_params+1) )
                    params_list[index] = params
                    attack_lb_list[:,:,index] = attack_lb.reshape((n_rows, n_cols))
                    attack_ub_list[:,:,index] = attack_ub.reshape((n_rows, n_cols))
                    # breakpoint()
                    # if np.abs(params[0] - -0.1750305) < 1e-3:
                    #     breakpoint()
                    # if len(np.nonzero(attack_ub-attack_lb)) > 1:
                    #     print(np.nonzero(attack_ub-attack_lb))
                    # print(index)
                    # attack_imgs.append((params, attack_lb, attack_ub))
                    # checked.append(False)


        tot_time += time.time() - begtime
    lb_biases = np.array(lexpr_cst).reshape((n_rows, n_cols, n_channels))
    ub_biases = np.array(uexpr_cst).reshape((n_rows, n_cols, n_channels))
    lb_weights = np.array(lexpr_weights).reshape((n_rows, n_cols, n_channels))
    ub_weights = np.array(uexpr_weights).reshape((n_rows, n_cols, n_channels))
    LB = spec_lb[dim + param_idx]
    UB = spec_ub[dim + param_idx]
    bound_tup = [lb_weights, ub_weights, lb_biases, ub_biases, LB, UB]
    if args.attack:
        for i in range(n_rows):
            for j in range(n_cols):
                vis_pixel_curves(i,j, bound_tup, image.reshape((n_rows, n_cols)))
                # vis_pixel_attack(i,j,bound_tup, image.reshape((n_rows, n_cols)), params_list, attack_lb_list, attack_ub_list)
    # if args.attack:
        # vis_pixel_rotations(bound_tup, image.reshape((n_rows, n_cols)), params_list, attack_lb_list, attack_ub_list)
    # view_linfun(0, bound_tup, image.reshape((n_rows, n_cols)), 20)


if __name__=="__main__":
    main()