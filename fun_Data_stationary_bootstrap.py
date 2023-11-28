
# %   Python code based on the Matlab code for stationary block bootstrap
# %
# %   Patton, A., D. Politis, and H. White (2009).
# %   Correction to: Automatic block-length selection for
# %   the dependent bootstrap. Econometric Reviews 28, 372â375.
# %
# %
# %   Politis, D. and J. Romano (1994). The stationary bootstrap.
# %   Journal of the American Statistical
# %   Association 89, 1303â1313.
# %
# %   Politis, D. and H. White (2004). Automatic block-length selection
# %   for the dependent bootstrap.
# %   Econometric Reviews 23, 53â70.

import numpy as np


def stationary_bootstrap( in_array, bootstrap_n_tot, bootstrap_exp_block_size, bootstrap_fixed_block):
    # return out_array, block_numbers

    #%  OBJECTIVE: generates one sample from time series, uses
    # %             stationary block bootstrap sampling
    # %
    # %             To carry out simulations, repeatedly call this function


    n_tot = int(bootstrap_n_tot)
    exp_block_size = int(bootstrap_exp_block_size)
    fixed_block = bootstrap_fixed_block

    # % INPUT:
    # %
    # %   in_array.shape = [row, col]: input data in numpy timeseries,
    #                       row = number of time pts, i.e. the actual time series values in rows
    # %                     col = number of data values, one column for each time series
    # %
    # %   n_tot:          total size of bootstrap sample
    # %                   n_tot <= row
    # %                   may be made up of many sub-blocks
    # %
    # %   exp_block_size:     integer expected subblock size
    # %                       should be  1 <= exp_block_size <= n_tot
    # %
    # %  fixed_block:  = False use stationary block bootstrap
    # %                             i.e. subblock size selected from a
    # %                             geometric distribution, with expected
    # %                             blocksize = exp_block_size
    # %                = True use fixed blocksize = exp_block_size


    # %  OUTPUT:
    # %
    # %    out_array.shape = [n_tot, col]: numpy array with output bootstrapped time series data
    # %        block bootstrapped resample of original data
    # %          size of resampled time series = n_tot
    # %          samples are simultaneously drawn from the array
    # %          of data values, i.e. each time pt out_array( index, :)
    # %          corresponds to some index2 in in_array(index2, :)
    #
    # %    block_numbers(n_blocks) = list of length n_blocks
    # %            col vector: block_numbers(j) SIZE of j'th block
    # %                        debugging info only

    row = in_array.shape[0]
    col = in_array.shape[1]

    out_array = np.zeros([n_tot, col])

    #% dummy values to ensure
    #% initial values set in loop

    sub_block_total = row
    actual_block_size = 0

    total_samples = 1

    current_block = 1

    block_numbers = []  #empty list

    while total_samples <= n_tot:
        if (sub_block_total > actual_block_size): #start subblock
            # restart sample
            # choose random starting index in [1,...,row]

            index = np.round(1 + np.random.random(1) * (row - 1))
            index = int(index[0])   #Extract just the number

            if fixed_block == False:    #START: assigning actual_block_size

                #%  choose subblock size from shifted geometric distribution
                #%  add one to matlab, since matlab is the unshifted
                #%  distribution, we need the shifted distribution
                actual_block_size = np.random.geometric(1/exp_block_size,1) + 1
                actual_block_size = actual_block_size[0] #extract just the number

            elif fixed_block == True:
                # fixed blocksize
                actual_block_size = exp_block_size

            #END: assigning actual_block_size

            sub_block_total = 1 #running count of number of samples in this subblock

            block_numbers.append(actual_block_size) #Append to output
            current_block = current_block + 1

        #END subblock


        # wrap around if necessary: CIRCULAR BOOTSTRAP
        if index > row:
            index = index - row


        out_array[total_samples-1, :] = in_array[index-1, :]
        index = index + 1
        sub_block_total = sub_block_total + 1
        total_samples = total_samples + 1;

    #END of while loop

    return out_array, block_numbers