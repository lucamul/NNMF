import math
import matplotlib.pyplot as plt

def calculate_flops():
    m = 1280
    n = 1280
    r = 1280
    num_iter = 5
    
    # Current flop count computation
    adds = num_iter * (5*m*n*r + m*n + r*m*r + r*r*n) + m*n*r
    mults = num_iter * (5*m*n*r + r*m*r + r*r*n + r*n) + m*n*r
    divs = num_iter * (m*r + r*n)
    
    # Current flop count init
    # flops_init = 2*m*n + n + r*q*m + r*m    
    flops_init = 0

    return mults + adds + divs + flops_init


def roofline():
    beta = 15/2.9
    max_perf = 4
    max_perf_128 = 16
    max_perf_256 = 32

    # memory accesses
    mabasic  = 244018438339
    maopt1  = 113520705519
    maopt2 = 195505064976
    maopt3  = 58951568494
    maopt4  = 130845025233
    maopt5  = 130733221879
    maopt6  = 48650059458
    maopt7  = 22621978639

    # cycles
    cbasic = 599913542977
    copt1 = 1130284699733
    copt2 = 1191900931170
    copt3 = 59382866774
    copt4 = 98750966230
    copt5 = 105165470873
    copt6 = 28756424700
    copt7 = 20976556225

    flop_count = calculate_flops()

    # performance
    pbasic = flop_count/cbasic
    popt1 = flop_count/copt1    
    popt2 = flop_count/copt2
    popt3 = flop_count/copt3
    popt4 = flop_count/copt4
    popt5 = flop_count/copt5
    popt6 = flop_count/copt6
    popt7 = flop_count/copt7

    # operational intensity
    ibasic = flop_count/mabasic
    iopt1 = flop_count/maopt1 
    iopt2 = flop_count/maopt2 
    iopt3 = flop_count/maopt3     
    iopt4 = flop_count/maopt4 
    iopt5 = flop_count/maopt5 
    iopt6 = flop_count/maopt6 
    iopt7 = flop_count/maopt7  

    op_int = max_perf/beta
    op_int_128 = max_perf_128/beta
    op_int_256 = max_perf_256/beta

    x_axis = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]

    boundary = []

    for x in x_axis:
        y_coord = (beta*x)
        boundary.append(y_coord)

    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['ytick.minor.width'] = 0

    fig,ax = plt.subplots(figsize=(12,7))

    # x_ticks = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
    # y_ticks = [0.07, 0.14, 0.28, 0.56, 1.12, 2.24, 4.48, 8.96, 17.92, 35.84]
    x_ticks = [0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
    y_ticks = [64 * (0.5**(10-i)) for i in range(11)]

    x_scalar = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, op_int, 2.56, 5.12, 10.24]
    y_scalar = [0.02*beta, 0.04*beta, 0.08*beta, 0.16*beta, 0.32*beta, 0.64*beta, max_perf, max_perf, max_perf, max_perf]
    x_vector128 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, op_int_128, 10.24]
    y_vector128 = [0.02*beta, 0.04*beta, 0.08*beta, 0.16*beta, 0.32*beta, 0.64*beta, 1.28*beta, 2.56*beta, max_perf_128, max_perf_128]
    x_vector256 = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, op_int_256, 10.24]
    y_vector256 = [0.02*beta, 0.04*beta, 0.08*beta, 0.16*beta, 0.32*beta, 0.64*beta, 1.28*beta, 2.56*beta, 5.12*beta, max_perf_256, max_perf_256]

    bound_scalar_x = [op_int, op_int]
    bound_scalar_y = [0.0625, max_perf]

    bound_vector128_x = [op_int_128, op_int_128]
    bound_vector128_y = [0.0625, max_perf_128]

    bound_vector256_x = [op_int_256, op_int_256]
    bound_vector256_y = [0.0625, max_perf_256]

    max_performance_scalar_x = [0.02, op_int]
    max_performance_scalar_y = [max_perf, max_perf]

    max_performance_vector128_x = [0.02, op_int_128]
    max_performance_vector128_y = [max_perf_128, max_perf_128]

    max_performance_vector256_x = [0.02, op_int_256]
    max_performance_vector256_y = [max_perf_256, max_perf_256]

    ax.scatter(ibasic, pbasic, s=20, label="Basic")
    ax.scatter(iopt1, popt1, s=20, label="Loop Gathering")
    ax.scatter(iopt2, popt2, s=20, label="ILP")
    ax.scatter(iopt3, popt3, s=20, label="Blocking for Cache")
    ax.scatter(iopt4, popt4, s=20, label="Blocking for Registers")
    #ax.scatter(iopt5, popt5, s=20, label="Blocking For Registers - ILP")
    ax.scatter(iopt6, popt6, s=20, label="Vectorization - 128")
    ax.scatter(iopt7, popt7, s=20, label="Vectorization - 256")

    ax.set_xscale('log')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    ax.set_yscale('log')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    ax.plot(x_scalar, y_scalar, label="Scalar Roofline")
    ax.plot(x_vector128, y_vector128, label="Vector 128 Roofline")
    ax.plot(x_vector256, y_vector256, label="Vector 256 Roofline")
    
    ax.plot(bound_scalar_x, bound_scalar_y, linestyle="dashed", label="Scalar Memory/Compute Bound")
    ax.plot(bound_vector128_x, bound_vector128_y, linestyle="dashed", label="Vector 128 Memory/Compute Bound")
    ax.plot(bound_vector256_x, bound_vector256_y, linestyle="dashed", label="Vector 256 Memory/Compute Bound")
    ax.plot(max_performance_scalar_x, max_performance_scalar_y, linestyle="dashed", label="Scalar Peak Performance")
    ax.plot(max_performance_vector128_x, max_performance_vector128_y, linestyle="dashed", label="Vector 128 Peak Performance")
    ax.plot(max_performance_vector256_x, max_performance_vector256_y, linestyle="dashed", label="Vector 256 Peak Performance")

    loc, labels = plt.xticks()
    plt.ylim(y_ticks[0], y_ticks[-1])
    plt.xlim(left=loc[0], right=loc[len(loc)-1])

    plt.title("Roofline Plot")
    plt.ylabel("P = W/T (flops/cycle)")
    plt.xlabel("I = W/Q (flops/byte)")

    plt.legend(bbox_to_anchor = (1.05, 0.6))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    fig.savefig('../plots/roofline.png')
    fig.savefig('../plots/roofline.eps')
    # plt.show()

roofline()
# %%
