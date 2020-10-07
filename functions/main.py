import os

import numpy as np
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg
from sklearn.neighbors import NearestNeighbors

from flask import jsonify
from google.cloud import storage

import pandas as pd
from io import BytesIO
from tensorflow.python.lib.io import file_io

import firebase_admin.auth

# this is required to read data from the storage
storage_client = storage.Client()

# initialise firebase
if not firebase_admin._apps:
    default_app = firebase_admin.initialize_app()


def splitall(path):
    # https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html

    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def read_lookuptable(bucket, filename):
    # https://stackoverflow.com/questions/49357352/read-csv-from-google-cloud-storage-to-pandas-dataframe
    return pd.read_csv(f"gs://{bucket}/{filename}")


def read_adj_matrix(bucket, filename):
    # https://stackoverflow.com/questions/44657902/how-to-load-numpy-npz-files-in-google-cloud-ml-jobs-or-from-google-cloud-storage
    f = BytesIO(file_io.read_file_to_string(f"gs://{bucket}/{filename}", binary_mode=True))
    loaded = np.load(f)

    matrix_format = loaded['format']

    matrix_format = matrix_format.item()

    if not isinstance(matrix_format, str):
        # Play safe with Python 2 vs 3 backward compatibility;
        # files saved with SciPy < 1.0.0 may contain unicode or bytes.
        matrix_format = matrix_format.decode('ascii')

    try:
        cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
    except AttributeError:
        raise ValueError('Unknown matrix format "{}"'.format(matrix_format))

    if matrix_format in ('csc', 'csr', 'bsr'):
        mat = cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    elif matrix_format == 'dia':
        mat = cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
    elif matrix_format == 'coo':
        mat = cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
    else:
        raise NotImplementedError('Load is not implemented for '
                                  'sparse matrix of format {}.'.format(matrix_format))

    return mat


def overlap(box_a, box_b):
    max_xy = np.minimum(box_a[:, :, 2:], box_b[:, :, 2:])
    min_xy = np.maximum(box_a[:, :, :2], box_b[:, :, :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])  # [A,B]

    return inter / area_a


def mask2lookup_table(mask, nrows, ncols, gridSize, wsi_highest_magnification, table):
    scaling_factor = wsi_highest_magnification / table.loc[0, 'magnification']
    gridSize_backend = table.loc[1, 'y'] - table.loc[0, 'y']

    # reconstruct frontend grid
    xv, yv = np.meshgrid(np.arange(ncols), np.arange(nrows))
    X = xv.ravel() * gridSize
    Y = yv.ravel() * gridSize

    # boxA = backend grid
    fromx = table['x'].values * scaling_factor
    fromy = table['y'].values * scaling_factor
    tox = (table['x'].values + gridSize_backend) * scaling_factor
    toy = (table['y'].values + gridSize_backend) * scaling_factor
    boxA = np.hstack([fromx[:, np.newaxis], fromy[:, np.newaxis], tox[:, np.newaxis], toy[:, np.newaxis]])

    XAcenter = (fromx + tox) / 2
    YAcenter = (fromy + toy) / 2

    # boxB = frontend grid
    fromx = X
    fromy = Y
    tox = (X + gridSize)
    toy = (Y + gridSize)
    boxB = np.hstack([fromx[:, np.newaxis], fromy[:, np.newaxis], tox[:, np.newaxis], toy[:, np.newaxis]])

    XBcenter = (fromx + tox) / 2
    YBcenter = (fromy + toy) / 2

    # for each backend cell (boxA) find the nearest frontend cell (boxB)
    n_neighbors = int(((gridSize_backend * scaling_factor/ gridSize) + 1) ** 2)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(np.hstack([YBcenter.reshape(-1, 1), XBcenter.reshape(-1, 1)]))
    _, indices = nbrs.kneighbors(np.hstack([YAcenter.reshape(-1, 1), XAcenter.reshape(-1, 1)]))

    # calculate intersect(A,B) / area(A)
    boxB = boxB[indices]  # [len(A), n_neighbors, 4]
    boxA = np.tile(boxA[:, np.newaxis, :], (1, n_neighbors, 1))  # [len(A), n_neighbors, 4]
    intersect_mat = overlap(boxA, boxB)  # [len(A), n_neighbours (B)]

    # using weighted sum to make a call for each cell of the backend grid
    labels = np.round((intersect_mat * mask[indices]).sum(axis=1)).astype(np.int)

    # labels = 0 -> null
    flag = labels == 0

    table['label'] = labels - 1  # unlabeled should be -1
    table.loc[np.logical_not(flag), 'manual'] = 1

    return table


def lookup_table2mask(nrows, ncols, gridSize, wsi_highest_magnification, table):
    scaling_factor = wsi_highest_magnification / table.loc[0, 'magnification']
    gridSize_backend = table.loc[1, 'y'] - table.loc[0, 'y']

    # reconstruct frontend grid
    xv, yv = np.meshgrid(np.arange(ncols), np.arange(nrows))
    X = xv.ravel() * gridSize
    Y = yv.ravel() * gridSize

    # boxA = frontend grid
    fromx = X
    fromy = Y
    tox = (X + gridSize)
    toy = (Y + gridSize)
    boxA = np.hstack([fromx[:, np.newaxis], fromy[:, np.newaxis], tox[:, np.newaxis], toy[:, np.newaxis]])

    XAcenter = (fromx + tox) / 2
    YAcenter = (fromy + toy) / 2

    # boxB = backend grid
    fromx = table['x'].values * scaling_factor
    fromy = table['y'].values * scaling_factor
    tox = (table['x'].values + gridSize_backend) * scaling_factor
    toy = (table['y'].values + gridSize_backend) * scaling_factor
    boxB = np.hstack([fromx[:, np.newaxis], fromy[:, np.newaxis], tox[:, np.newaxis], toy[:, np.newaxis]])

    XBcenter = (fromx + tox) / 2
    YBcenter = (fromy + toy) / 2

    # for each frontend cell (boxA) find the nearest backend cell (boxB)
    n_neighbors = int((gridSize / (gridSize_backend * scaling_factor) + 1) ** 2)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(np.hstack([YBcenter.reshape(-1, 1), XBcenter.reshape(-1, 1)]))
    _, indices = nbrs.kneighbors(np.hstack([YAcenter.reshape(-1, 1), XAcenter.reshape(-1, 1)]))

    # calculate intersect(A,B) / area(A)
    boxB = boxB[indices]  # [len(A), n_neighbors, 4]
    boxA = np.tile(boxA[:, np.newaxis, :], (1, n_neighbors, 1))  # [len(A), n_neighbors, 4]
    intersect_mat = overlap(boxA, boxB)  # [len(A), n_neighbours (B)]

    # using weighted sum to make a call for each cell of the backend grid
    labels = table['label'].values
    labels = labels + 1  # values start from 0
    mask = (np.round((intersect_mat * labels[indices]).sum(axis=1))).astype(np.int)

    return mask


def label_propagation(Wn, labeled_indices, labels, nclasses, alpha=0.01, max_iter=20):
    """
    :param Wn: normalised adjacency matrix
    :param labeled_indices: a list of indices of nodes with manual labels
    :param labels: a list of manual labels
    :param nclasses: the number of classes
    :param alpha: [0,1] how much you trust the manual label
    :param max_iter: the number of iterations in conjugate gradient descent
    :return: pseudo_label:
    :return: weights:
    """

    assert isinstance(labels, np.ndarray), 'labels must be a numpy array'
    assert isinstance(labeled_indices, np.ndarray), 'labeled_indices must be a numpy array'

    N = Wn.shape[0]

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, nclasses))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(nclasses):
        cur_idx = labeled_indices[labels == i]  # check this !!!!!
        if len(cur_idx):
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    ########################################################
    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    row_sums = Z.sum(axis=1)
    probs_l1 = Z / row_sums[:, np.newaxis]
    probs_l1[np.isnan(probs_l1)] = 0
    probs_l1[probs_l1 < 0] = 0

    if nclasses > 1:
        entropy = scipy.stats.entropy(probs_l1.T)
        entropy[np.isnan(entropy)] = np.log(nclasses)  # there is a possibility that the entropy value will be NaN

        weights = 1 - entropy / np.log(nclasses)
        weights = weights / np.max(weights)

    else:
        weights = probs_l1 / np.max(probs_l1)
        weights = weights.ravel()

    pseudo_labels = np.argmax(probs_l1, 1)
    pseudo_labels[weights == 0] = -1  # if weight == 0, it's useless

    pseudo_labels[labeled_indices] = labels  # check this!!!
    weights[labeled_indices] = 1.0

    return pseudo_labels, weights


def propagate_label(request):
    """
    :param request: must be a json containing ('bucket', 'image', 'nCols', 'nRows', 'gridSize', 'mask', 'idToken')
    :return:
    """
    if request.method != 'POST':
        return 'Only POST requests are accepted', 405

    # request json
    request_json = request.get_json(force=True)

    # check the token is valid
    if 'idToken' not in request_json:
        return "the json does not contain 'idToken'"

    id_token = request_json['idToken']

    try:
        firebase_admin.auth.verify_id_token(id_token)
    except:
        return "The request is not authorised.", 403

    # check other required fields
    if 'bucket' not in request_json:
        return "the json does not contain 'bucket'"
    elif 'image' not in request_json:
        return "the json does not contain 'image'"
    elif 'nCols' not in request_json:
        return "the json does not contain 'nCols'"
    elif 'nRows' not in request_json:
        return "the json does not contain 'nRows'"
    elif 'gridSize' not in request_json:
        return "the json does not contain 'gridSize'"
    elif 'mask' not in request_json:
        return "the json does not contain 'mask'"
    else:
        # Get a reference to the destination file in GCS
        bucket_name = request_json['bucket']
        image_blob = request_json['image']

        # propose to change this to store in firebase
        # 1. the location of paths
        # 2. the nominal objective
        blob_path = splitall(image_blob)[0:-2]
        table_blob = os.path.join(*blob_path, 'anno', 'lookup_table.csv')
        matrix_blob = os.path.join(*blob_path, 'anno', 'adj_mat.npz')
        wsi_highest_magnification = 40.0

        # loading lookup table and adjacency matrix
        lookup_table = read_lookuptable(bucket_name, table_blob)
        Wn = read_adj_matrix(bucket_name, matrix_blob)

        # matching label from mask to the lookup table
        mask = np.array(eval(request_json['mask']))
        nrows = request_json['nRows']
        ncols = request_json['nCols']
        gridSize = request_json['gridSize']
        lookup_table = mask2lookup_table(mask, nrows, ncols, gridSize, wsi_highest_magnification, lookup_table)

        # only selected tiles with manual label
        flag = lookup_table['manual'].values == 1
        labels = lookup_table['label'].values[flag]
        labeled_indices = lookup_table['label'].index.values[flag]
        nclasses = len(np.unique(labels))
        pseudo_labels, _ = label_propagation(Wn, labeled_indices, labels, nclasses)
        lookup_table['label'] = pseudo_labels

        # matching pseudo labels with the frontend grid
        mask = lookup_table2mask(nrows, ncols, gridSize, wsi_highest_magnification, lookup_table)

        return jsonify({'mask': mask.tolist()})
